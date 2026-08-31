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
)
from forge.contracts.termination import MaxStepsTerminationPolicy
from forge.contracts.types import Action, ActionResult
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


class _HttpInitialState(InitialStateProvider):
    """Resets a container env over HTTP: POST /forge/reset (seeded when a
    seed is given, unseeded otherwise resets to the app's fixed baseline),
    then GET /forge/state for the resulting state. ``ContainerEnvBase.reset``
    delegates here so there is exactly one place that knows how to reset a
    container environment.
    """

    def __init__(self, base_url: str, client: httpx.Client) -> None:
        self._base_url = base_url
        self._client = client

    def reset(
        self, ctx: "RuntimeContext", *, seed: int | None, options
    ) -> dict:
        json_body = {"seed": seed} if seed is not None else None
        response = self._client.post(f"{self._base_url}/forge/reset", json=json_body)
        response.raise_for_status()
        state_response = self._client.get(f"{self._base_url}/forge/state")
        state_response.raise_for_status()
        return state_response.json()


class _PassthroughObservationEncoder(ObservationEncoder):
    """Container state already IS the observation (GET /forge/state, see
    ``_observe``); this stub satisfies the facade without duplicating that
    HTTP call in-process."""

    def encode(self, state: dict, ctx: "RuntimeContext") -> Observation:
        return Observation(payload=state)


class _HttpActionResult(ActionResult):
    """``ActionResult`` plus the raw HTTP response.

    ``ExecutionBackend.execute`` is only contracted to return an
    ``ActionResult``, and this is a genuine subtype of one — every consumer
    that only knows about ``ActionResult`` still works. The extra field lets
    ``step`` retrieve a real ``httpx.Response`` for ``compute_reward`` without
    a shared, stateful side channel: each call gets its own result, so two
    interleaved ``execute`` calls never race for it the way a `last_response`
    attribute on the backend would.

    ``exclude=True`` keeps `response` out of `model_dump()` /
    `model_dump(mode="json")` / `model_dump_json()` — an `httpx.Response`
    isn't JSON-serializable, and nothing that logs or replays an
    `ActionResult` generically should have to know this one family attaches
    something extra. The field itself stays a normal attribute, so `.response`
    is still readable in-process by `step`.
    """

    model_config = {"arbitrary_types_allowed": True}

    response: httpx.Response = Field(exclude=True)


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
    """

    def __init__(
        self,
        base_url: str,
        client: httpx.Client,
        endpoint_for: Callable[[dict], str],
    ) -> None:
        self._base_url = base_url
        self._client = client
        self._endpoint_for = endpoint_for

    def execute(self, action: Action, state: dict, ctx: "RuntimeContext") -> ActionResult:
        action_dict = action.to_dict()
        endpoint = self._endpoint_for(action_dict)
        response = self._client.post(f"{self._base_url}{endpoint}", json=action_dict)
        state_response = self._client.get(f"{self._base_url}/forge/state")
        state_response.raise_for_status()
        return _HttpActionResult(state=state_response.json(), response=response)


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
        self._initial_state = _HttpInitialState(self.base_url, self.client)
        self._observations = _PassthroughObservationEncoder()
        self._backend = _HttpExecutionBackend(
            self.base_url, self.client, self.action_endpoint
        )
        self._rubric = TaskSuccessRubric()
        self._termination = MaxStepsTerminationPolicy(max_steps=max_steps)

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

    def compute_reward(self, response: httpx.Response, obs: dict) -> float:
        return 1.0 if response.status_code == 200 else 0.0

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    def _observe(self) -> dict:
        response = self.client.get(f"{self.base_url}/forge/state")
        response.raise_for_status()
        return response.json()

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        # Thread the seed to the app so the same seed reproduces the same
        # starting universe and different seeds yield different-but-reproducible
        # ones. An unseeded reset resets to the app's fixed baseline. Delegates
        # to `self.initial_state` — see `_HttpInitialState` — so there is
        # exactly one place that knows how to reset a container environment.
        obs = self._initial_state.reset(None, seed=seed, options=options or {})
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
        return result.state, reward, False, False, {
            "status_code": result.response.status_code
        }

    def close(self) -> None:
        self.client.close()
