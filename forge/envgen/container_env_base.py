"""Base class for container-backed environments.

LLM-generated environment packages used to regenerate ~100 lines of identical
HTTP plumbing per environment. All of it lives here instead: the generator
only subclasses and overrides the two domain-specific hooks, which makes
generated packages smaller, faster to produce, and impossible to get wrong in
the plumbing.
"""
from __future__ import annotations

import gymnasium
import httpx


class ContainerEnvBase(gymnasium.Env):
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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.observation_space = gymnasium.spaces.Dict({})
        self.action_space = gymnasium.spaces.Dict({})

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
        response = self.client.post(f"{self.base_url}/forge/reset")
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
