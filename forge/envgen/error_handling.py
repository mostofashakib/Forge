from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from forge.runtime.error_handling import get_error_handler
from forge.runtime.errors import ForgeError


@dataclass(frozen=True)
class GenerationErrorRecord:
    task_id: str
    agent_id: str
    error_type: str
    message: str
    created_at: datetime


class AgentExecutionError(ForgeError, RuntimeError):
    default_code = "AGENT_EXECUTION_FAILED"
    origin = "environment_generation"


class GenerationErrorHandler:
    """Normalizes and tracks specialist failures without swallowing causes."""

    origin = "environment_generation"

    def __init__(self) -> None:
        self._records: list[GenerationErrorRecord] = []

    def capture(self, *, task_id: str, agent_id: str, error: Exception) -> AgentExecutionError:
        record = GenerationErrorRecord(
            task_id=task_id,
            agent_id=agent_id,
            error_type=type(error).__name__,
            message=str(error),
            created_at=datetime.now(timezone.utc),
        )
        self._records.append(record)
        get_error_handler().capture(
            error,
            origin=self.origin,
            code=AgentExecutionError.default_code,
            context={"task_id": task_id, "agent_id": agent_id},
            reraise=False,
        )
        return AgentExecutionError(
            f"Agent {agent_id!r} failed task {task_id!r}: {error}",
            cause=error,
        )

    @property
    def records(self) -> tuple[GenerationErrorRecord, ...]:
        return tuple(self._records)
