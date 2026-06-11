from __future__ import annotations
from pydantic import BaseModel

from forge.runtime.errors import InvalidActionError

__all__ = [
    "InvalidActionError",
    "ToolParam",
    "ToolSpec",
    "EnvironmentSpec",
    "StepSnapshot",
]


class ToolParam(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class ToolSpec(BaseModel):
    """Schema describing one tool an agent may call — the env's tool use surface."""

    name: str
    description: str = ""
    params: list[ToolParam] = []


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
