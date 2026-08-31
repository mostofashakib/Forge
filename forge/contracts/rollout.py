"""The canonical record exchanged by collectors, exporters, and trainers."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RolloutOutcome = Literal["success", "failure", "partial_success", "edge_case"]


class RolloutRecord(BaseModel):
    """One completed policy interaction with an environment.

    Environment-specific controllers may retain richer result objects, but they
    all convert to this shape at the collection boundary. Training loaders use
    the same type, avoiding parallel, subtly incompatible rollout models.
    """

    episode_id: str
    env_name: str = ""
    task_name: str = ""
    prompt: str = ""
    completion: str = ""
    seed: int | None = None
    total_reward: float = 0.0
    per_step_rewards: list[float] = Field(default_factory=list)
    passed: bool = False
    outcome: RolloutOutcome = "failure"
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    invalid_actions: int = 0
    error: str | None = None
    behavior_model: str = ""
    termination_reason: str = "unknown"
    verification_results: list[dict] = Field(default_factory=list)
    reward_breakdown: dict = Field(default_factory=dict)
