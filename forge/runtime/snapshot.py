from __future__ import annotations
from pydantic import BaseModel

from forge.contracts.types import ToolParam, ToolSpec  # noqa: F401
from forge.runtime.errors import InvalidActionError

__all__ = [
    "InvalidActionError",
    "ToolParam",
    "ToolSpec",
    "EnvironmentSpec",
    "StepSnapshot",
]


class EnvironmentSpec(BaseModel):
    name: str
    domain: str
    max_steps: int = 50
    default_task: dict | None = None


class StepSnapshot(BaseModel):
    episode_id: str
    step_index: int
    state_hash_before: str
    state_hash_after: str
    action: dict
    events: list[dict]
    reward: float
    verifier_results: list[dict]
    diff: dict
    terminated: bool
    truncated: bool
