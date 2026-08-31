"""Base class for container-backed environments.

LLM-generated environment packages used to regenerate ~100 lines of identical
HTTP plumbing per environment. All of it lives here instead: the generator
only subclasses and overrides the two domain-specific hooks, which makes
generated packages smaller, faster to produce, and impossible to get wrong in
the plumbing.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import gymnasium
import httpx
from pydantic import Field

from forge.contracts import (
    Environment,
    ExecutionBackend,
    InitialStateProvider,
    Observation,
    ObservationEncoder,
    Rubric,
    StateManager,
    Task,
    TaskSource,
    TerminationPolicy,
    Transport,
    TransportRequest,
    TransportResponse,
)
from forge.contracts.termination import MaxStepsTerminationPolicy
from forge.contracts.types import Action, ActionResult, StepOutcome
from forge.runtime.http_state import HttpStateManager
from forge.runtime.reward import TaskSuccessRubric
from forge.runtime.rest_transport import RestTransport

if TYPE_CHECKING:
    from forge.runtime.context import RuntimeContext


class _NoTaskSource(TaskSource):
    """Container envs have no task registry yet; generated subclasses that
    gain one can replace this via ``self._task_source``."""

    def tasks(self) -> Sequence[Task]:
        return ()

    def get(self, task_id: str) -> Task:
        raise KeyError(task_id)


class ContainerTransportError(RuntimeError):
    """A container HTTP call failed.

    ``RestTransport`` reports failures in-band so one bad call costs a step
    rather than the episode. This family's callers have always seen an
    exception instead, so ``_raise_for`` converts an in-band error back into
    a raise at the boundary, and this is what they now see. It carries the
    status when there was one — ``0`` means the request never completed.
    """

    def __init__(self, target: str, status: int, detail: str) -> None:
        super().__init__(f"{target} failed (status {status}): {detail}")
        self.target = target
        self.status = status


def _call(transport: Transport, method: str, target: str, payload: dict | None = None):
    """Make one call and raise if it did not succeed.

    Every container call goes through here, so the in-band-to-raise
    conversion lives in exactly one place, and every call gets the
    transport's timeout and JSON-decode handling.
    """
    # An empty payload is how TransportRequest spells "no body"; RestTransport
    # maps it back to `json=None` on the wire, which is what these calls sent
    # before. `None` is not a valid payload, so it must not be passed through.
    response = transport.call(
        TransportRequest(method=method, target=target, payload=payload or {})
    )
    _raise_for(response, target)
    return response


def _raise_for(response: TransportResponse, target: str) -> None:
    if response.error:
        raise ContainerTransportError(target, response.status, response.error)
    if not 200 <= response.status < 300:
        # Matches the `raise_for_status()` these paths used to call.
        raise ContainerTransportError(target, response.status, "HTTP error")


class _HttpInitialState(InitialStateProvider):
    """Resets a container env over HTTP: POST /forge/reset (seeded when a
    seed is given, unseeded otherwise resets to the app's fixed baseline),
    then GET /forge/state for the resulting state. ``ContainerEnvBase.reset``
    delegates here so there is exactly one place that knows how to reset a
    container environment.

    Goes through the env's own ``RestTransport`` rather than holding a second
    ``httpx.Client``, so the transport's timeout and JSON-decode handling
    protect this path — an unset timeout reaching httpx as ``None`` means no
    timeout at all, and a proxy's HTML 502 in place of JSON is a real failure
    mode for a container that is still starting. A failed reset still raises,
    as it always has; see ``ContainerTransportError``.
    """

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def reset(
        self, ctx: "RuntimeContext", *, seed: int | None, options
    ) -> dict:
        json_body = {"seed": seed} if seed is not None else None
        _call(self._transport, "POST", "/forge/reset", json_body)
        return _call(self._transport, "GET", "/forge/state").body


class _PassthroughObservationEncoder(ObservationEncoder):
    """Container state already IS the observation (GET /forge/state, see
    ``_observe``); this stub satisfies the facade without duplicating that
    HTTP call in-process."""

    def encode(self, state: dict, ctx: "RuntimeContext") -> Observation:
        return Observation(payload=state)


class _ActionResponse:
    """What ``compute_reward`` receives for the action call.

    The hook is public API that generated subclasses override, and the
    prompt that generates them types the parameter as bare ``response``. It
    has only ever been read for ``status_code``, so this exposes that plus
    the decoded body, and keeps working for every existing override while
    the underlying call moves to the transport.
    """

    __slots__ = ("status_code", "body")

    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self.body = body

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self.body


class _HttpActionResult(ActionResult):
    """``ActionResult`` plus the action call's response.

    ``ExecutionBackend.execute`` is only contracted to return an
    ``ActionResult``, and this is a genuine subtype of one — every consumer
    that only knows about ``ActionResult`` still works. The extra field lets
    ``step`` retrieve the response for ``compute_reward`` without a shared,
    stateful side channel: each call gets its own result, so two interleaved
    ``execute`` calls never race for it the way a `last_response` attribute
    on the backend would.

    ``exclude=True`` keeps `response` out of `model_dump()` /
    `model_dump(mode="json")` / `model_dump_json()` — nothing that logs or
    replays an `ActionResult` generically should have to know this one
    family attaches something extra. The field itself stays a normal
    attribute, so `.response` is still readable in-process by `step`.
    """

    model_config = {"arbitrary_types_allowed": True}

    response: _ActionResponse = Field(exclude=True)


class _HttpExecutionBackend(ExecutionBackend):
    """Executes a container env's actions over HTTP: POST to the endpoint
    ``endpoint_for`` resolves for the action (bound to the env's own
    ``action_endpoint`` hook, so a subclass override is honored), then GET
    /forge/state for the resulting state. ``ContainerEnvBase.step`` delegates
    here so there is exactly one place that knows how to execute an action.

    Takes a real ``Action``, per the ``ExecutionBackend`` contract, and
    converts once at this boundary — ``action.to_dict()`` — for the two
    things that need the wire form: the ``action_endpoint`` hook (which stays
    dict-based, since it is the public hook generated subclasses override)
    and the JSON POST body.

    Goes through the env's own ``RestTransport``, like ``_HttpInitialState``,
    so both calls get its timeout and JSON-decode handling.

    The two calls fail differently, and deliberately. A non-2xx on the ACTION
    post does NOT raise: an action the app rejects has always cost the step's
    reward rather than the episode, and ``compute_reward`` is given the status
    to score. A wire failure on that post, or any failure fetching state
    afterwards, does raise — the episode cannot continue without knowing what
    the state is.
    """

    def __init__(
        self,
        transport: Transport,
        endpoint_for: Callable[[dict], str],
    ) -> None:
        self._transport = transport
        self._endpoint_for = endpoint_for

    def execute(self, action: Action, state: dict, ctx: "RuntimeContext") -> ActionResult:
        action_dict = action.to_dict()
        endpoint = self._endpoint_for(action_dict)
        response = self._transport.call(
            TransportRequest(method="POST", target=endpoint, payload=action_dict)
        )
        # A rejected action is scored, not raised — but a call that never
        # completed leaves nothing to score, so that still raises.
        if response.error:
            raise ContainerTransportError(endpoint, response.status, response.error)
        state_response = _call(self._transport, "GET", "/forge/state")
        return _HttpActionResult(
            state=state_response.body,
            response=_ActionResponse(response.status, response.body),
        )


class ContainerEnvBase(gymnasium.Env, Environment):
    """Gymnasium env wrapping a containerized FastAPI app over HTTP.

    Provides everything common to container environments:
      reset()  → POST {base_url}/forge/reset, then observe
      step()   → POST {base_url}{action_endpoint(action)} with the action as
                 JSON, then observe and reward
      observe  → GET  {base_url}/forge/state (the SQLite-backed source of truth)

    Subclasses override only the domain-specific hooks:
      action_endpoint(action) — map an action dict to its endpoint
                                (default: "/{action['type']}")
      compute_reward(response, obs) — score a step
                                (default: 1.0 on HTTP 200, else 0.0)

    `transport` is the env's single I/O path: `_HttpInitialState` and
    `_HttpExecutionBackend` both run through it, so `reset()` and `step()`
    get its timeout and JSON-decode handling. It reports failures in-band;
    they are converted back to a raise at the boundary, so callers still see
    an exception. See `ContainerTransportError`.

    One facade member is exposed for contract compliance but does not sit on
    the hot path, and that is intentional rather than an oversight:
      rubric      — see the `rubric` property below. `step()`'s reward comes
                    from the public `compute_reward()` hook, not from
                    `self._rubric`; that hook is documented API that
                    generated subclasses override, and stays that way.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
        max_steps: int = 50,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.observation_space = gymnasium.spaces.Dict({})
        self.action_space = gymnasium.spaces.Dict({})

        # A container env genuinely has a transport and a state manager: it is
        # over a wire and its SQLite is the source of truth. `initial_state`
        # and `backend` are real HTTP collaborators too — `reset`/`step` below
        # delegate to them rather than duplicating their HTTP calls. Only
        # `task_source` and `observations` have no equivalent for this family
        # yet, so those two stay minimal stubs.
        self._state_manager = HttpStateManager(self.base_url, client=self.client)
        self._transport = RestTransport(self.base_url, client=self.client)
        self._task_source = _NoTaskSource()
        # Both HTTP collaborators run through the transport rather than
        # holding their own client, so its timeout and JSON-decode handling
        # protect the paths reset() and step() actually take.
        self._initial_state = _HttpInitialState(self._transport)
        self._observations = _PassthroughObservationEncoder()
        self._backend = _HttpExecutionBackend(self._transport, self.action_endpoint)
        self._rubric = TaskSuccessRubric()
        self._termination = MaxStepsTerminationPolicy(max_steps=max_steps)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Environment facade
    # ------------------------------------------------------------------

    @property
    def state(self) -> StateManager:
        return self._state_manager

    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def task_source(self) -> TaskSource:
        return self._task_source

    @property
    def initial_state(self) -> InitialStateProvider:
        return self._initial_state

    @property
    def observations(self) -> ObservationEncoder:
        return self._observations

    @property
    def backend(self) -> ExecutionBackend:
        return self._backend

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    @property
    def termination(self) -> TerminationPolicy:
        return self._termination

    # ------------------------------------------------------------------
    # Domain hooks
    # ------------------------------------------------------------------

    def action_endpoint(self, action: dict) -> str:
        return f"/{action['type']}"

    def compute_reward(self, response: _ActionResponse, obs: dict) -> float:
        return 1.0 if response.status_code == 200 else 0.0

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    def _observe(self) -> dict:
        return _call(self._transport, "GET", "/forge/state").body

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        # Thread the seed to the app so the same seed reproduces the same
        # starting universe and different seeds yield different-but-reproducible
        # ones. An unseeded reset resets to the app's fixed baseline. Delegates
        # to `self.initial_state` — see `_HttpInitialState` — so there is
        # exactly one place that knows how to reset a container environment.
        obs = self._initial_state.reset(None, seed=seed, options=options or {})
        self._step_count = 0
        return obs, {}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        # Delegates to `self.backend` — see `_HttpExecutionBackend` — so there
        # is exactly one place that knows how to execute an action against a
        # container environment. Converts at the boundary, the way
        # TransitionEngine.apply converts with Action.from_dict before
        # calling a handler, so the backend receives the typed value its own
        # contract declares.
        result = self._backend.execute(Action.from_dict(action), {}, None)
        reward = self.compute_reward(result.response, result.state)

        # Honor `self._termination` (a `MaxStepsTerminationPolicy`) instead
        # of hardcoding `False, False` — the budget `max_steps` describes is
        # otherwise wired up and never consulted, so a caller passing
        # `max_steps=10` got no truncation at all. `step_index` is the index
        # of *this* step (0-based, matching `MaxStepsTerminationPolicy`'s
        # `step_index >= max_steps - 1`), so it is read before incrementing.
        step_index = self._step_count
        self._step_count += 1
        termination = self._termination.check(
            StepOutcome(step_index=step_index, reward=reward)
        )
        # `MaxStepsTerminationPolicy` only ever returns `truncated=True`
        # (a budget ran out, not a natural end), but the mapping below stays
        # correct if `self._termination` is ever swapped for a policy that
        # signals a true terminal condition (`truncated=False`) instead.
        truncated = bool(termination) and termination.truncated
        terminated = bool(termination) and not termination.truncated

        return result.state, reward, terminated, truncated, {
            "status_code": result.response.status_code
        }

    def close(self) -> None:
        self.client.close()
