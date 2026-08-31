"""Base class for container-backed environments.

LLM-generated environment packages used to regenerate ~100 lines of identical
HTTP plumbing per environment. All of it lives here instead: the generator
only subclasses and overrides the two domain-specific hooks, which makes
generated packages smaller, faster to produce, and impossible to get wrong in
the plumbing.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import gymnasium
import httpx

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


class _NoInitialState(InitialStateProvider):
    """Reset happens over HTTP via POST /forge/reset (see ``reset``); this
    stub satisfies the facade without duplicating that logic in-process."""

    def reset(
        self, ctx: "RuntimeContext", *, seed: int | None, options
    ) -> dict:
        return {}


class _PassthroughObservationEncoder(ObservationEncoder):
    """Container state already IS the observation (GET /forge/state, see
    ``_observe``); this stub satisfies the facade without duplicating that
    HTTP call in-process."""

    def encode(self, state: dict, ctx: "RuntimeContext") -> Observation:
        return Observation(payload=state)


class _HttpExecutionBackend(ExecutionBackend):
    """Actions execute over HTTP via the POST in ``step``; this stub
    satisfies the facade without duplicating that logic in-process."""

    def execute(
        self, action: Action, state: dict, ctx: "RuntimeContext"
    ) -> ActionResult:
        return ActionResult(state=state)


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
        # over a wire and its SQLite is the source of truth. The remaining
        # collaborators have no in-process equivalent for this family yet, so
        # they are minimal stubs that satisfy the facade without duplicating
        # the HTTP plumbing below.
        self._state_manager = HttpStateManager(self.base_url, client=self.client)
        self._transport = RestTransport(self.base_url, client=self.client)
        self._task_source = _NoTaskSource()
        self._initial_state = _NoInitialState()
        self._observations = _PassthroughObservationEncoder()
        self._backend = _HttpExecutionBackend()
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
        # ones. An unseeded reset resets to the app's fixed baseline.
        json_body = {"seed": seed} if seed is not None else None
        response = self.client.post(f"{self.base_url}/forge/reset", json=json_body)
        response.raise_for_status()
        return self._observe(), {}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        response = self.client.post(
            f"{self.base_url}{self.action_endpoint(action)}", json=action
        )
        obs = self._observe()
        reward = self.compute_reward(response, obs)
        return obs, reward, False, False, {"status_code": response.status_code}

    def close(self) -> None:
        self.client.close()
