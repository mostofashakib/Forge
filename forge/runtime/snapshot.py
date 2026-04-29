from __future__ import annotations
from pydantic import BaseModel


class InvalidActionError(Exception):
    def __init__(self, detail: str, code: str = "INVALID_ACTION") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict:
        return {"error": "INVALID_ACTION", "code": self.code, "detail": self.detail}


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
