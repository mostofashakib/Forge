"""Shared API models used by environments, agents, and verifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class User:
    user_id: str
    display_name: str
    email: str
    role: str
    team: str
    handle: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", self.user_id)



@dataclass(frozen=True)
class ErrorState:
    error_code: str
    error_type: str
    message: str
    tool_name: str
    input_payload: dict[str, Any]
    retryable: bool
    virtual_timestamp: int
    state_changed: bool = False


@dataclass(frozen=True)
class Observation:
    observation_id: str
    environment_name: str
    virtual_timestamp: int
    payload: dict[str, Any]
    error: ErrorState | None = None


@dataclass(frozen=True)
class Action:
    tool_name: str
    input_payload: dict[str, Any]
    acting_user_id: str


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    input_payload: dict[str, Any]
    acting_user_id: str
    virtual_timestamp: int


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    output: dict[str, Any] | None
    error: ErrorState | None
    state_changed: bool
    virtual_timestamp: int


@dataclass(frozen=True)
class StateTransition:
    transition_id: str
    tool_name: str
    summary: str
    before_hash: str
    after_hash: str
    virtual_timestamp: int


@dataclass(frozen=True)
class HarborEvent:
    event_id: str
    event_type: str
    virtual_timestamp: int
    payload: dict[str, Any]


@dataclass
class Trajectory:
    task_id: str
    environment_name: str
    seed: int
    instruction: str
    observations: list[Observation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_outputs: list[ToolResult] = field(default_factory=list)
    errors: list[ErrorState] = field(default_factory=list)
    state_transitions: list[StateTransition] = field(default_factory=list)
    verifier_outputs: list[dict[str, Any]] = field(default_factory=list)
    harbor_events: list[HarborEvent] = field(default_factory=list)
    final_state_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class VerificationResult:
    score: Literal[0, 1]
    layer: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
