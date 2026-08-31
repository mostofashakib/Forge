"""Data types the contract interfaces speak in.

These are shapes, not behavior. They live here rather than in forge/runtime/
so that both the in-process and container environment families can depend on
them without either depending on the other.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """One problem the model should solve.

    Unifies the compiler's TaskTemplate and envgen's Scenario. `objective` is
    the natural-language goal; CLI and browser environments carry only that.
    """

    id: str
    objective: str
    seed: int | None = None
    success_conditions: list[dict] = Field(default_factory=list)
    failure_conditions: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Actions and observations
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """One action the model takes.

    The runtime's public surface still accepts plain dicts; engines convert at
    the boundary via `from_dict` so handler code receives a typed value.
    """

    type: str
    params: dict = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "Action":
        params = {k: v for k, v in raw.items() if k != "type"}
        return cls(type=raw["type"], params=params)

    def to_dict(self) -> dict:
        return {"type": self.type, **self.params}


class ActionResult(BaseModel):
    """What an execution backend returns after running one action."""

    state: dict
    events: list[dict] = Field(default_factory=list)
    error: dict | None = None


class Observation(BaseModel):
    """What the model sees back after an action.

    Carries all three shapes the families produce: a structured payload (gym
    dict, /forge/state), rendered text, and typed blocks for tool output.
    """

    payload: dict = Field(default_factory=dict)
    text: str | None = None
    blocks: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

class StepOutcome(BaseModel):
    """Everything a termination policy is allowed to decide on."""

    step_index: int
    score: float = 0.0
    reward: float = 0.0
    state_hash: str | None = None
    verifier_results: list["VerificationResult"] = Field(default_factory=list)


class Termination(BaseModel):
    """A decision to end the episode."""

    reason: str
    truncated: bool = False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ToolParam(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class ToolSpec(BaseModel):
    """Schema describing one tool an agent may call — the env's tool surface."""

    name: str
    description: str = ""
    params: list[ToolParam] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Verification and reward
# ---------------------------------------------------------------------------

class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float
    evidence: str | None = None


class VerificationResult(BaseModel):
    verifier_id: str
    passed: bool
    score: float
    checks: list[CheckResult]
    explanation: str = ""

    @classmethod
    def from_checks(
        cls, verifier_id: str, checks: list[CheckResult]
    ) -> "VerificationResult":
        passed = all(c.passed for c in checks)
        score = sum(c.score for c in checks) / len(checks) if checks else 0.0
        return cls(verifier_id=verifier_id, passed=passed, score=score, checks=checks)


class RewardComponent(BaseModel):
    name: str
    value: float


class RewardBreakdown(BaseModel):
    total_reward: float
    components: list[RewardComponent]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentAdapter(Protocol):
    """Anything that can pick an action given an observation."""

    def act(self, obs: dict, action_types: frozenset[str]) -> dict: ...


StepOutcome.model_rebuild()
