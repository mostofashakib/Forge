from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


class MessageKind(StrEnum):
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    ARTIFACT_AVAILABLE = "artifact_available"
    REVIEW_COMPLETED = "review_completed"


@dataclass(frozen=True)
class AgentMessage:
    """Transport envelope for communication between generation agents."""

    sender: str
    recipient: str
    kind: MessageKind
    task_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class A2AProtocol:
    """In-process A2A transport with task-scoped context permissions.

    Messages carry coordination metadata. Large generated artifacts stay on the
    artifact bus and are exposed only through the recipient's declared inputs.
    """

    def __init__(self) -> None:
        self._messages: list[AgentMessage] = []
        self._read_scopes: dict[str, frozenset[str]] = {}

    def register_scope(self, task_id: str, readable_artifacts: set[str]) -> None:
        self._read_scopes[task_id] = frozenset(readable_artifacts)

    def send(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def messages_for(self, recipient: str) -> tuple[AgentMessage, ...]:
        return tuple(
            message
            for message in self._messages
            if message.recipient in {recipient, "*"}
        )

    def context_for(self, task_id: str, artifacts: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = self._read_scopes.get(task_id, frozenset())
        return MappingProxyType(
            {name: value for name, value in artifacts.items() if name in allowed}
        )

    @property
    def history(self) -> tuple[AgentMessage, ...]:
        return tuple(self._messages)
