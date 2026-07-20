"""Unified per-agent-run logger.

Captures the agent's LLM-call layer (prompt, chosen tool call, raw response)
alongside tool/action invocations and state changes as one structured, ordered,
queryable trace, correlated by run id and step. This complements
``TrajectoryStore``/``StepSnapshot`` (which feed the verifiers) by also recording
the reasoning/tool-call layer the trajectory does not; it does not duplicate
their storage.

The trace is the structured input the per-run failure-mode analysis consumes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

RUN_START = "run_start"
LLM_CALL = "llm_call"
ACTION = "action"
STATE_CHANGE = "state_change"
RUN_END = "run_end"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LogEntry(BaseModel):
    """One ordered event in an agent run's trace."""

    seq: int
    run_id: str
    kind: str
    ts: str
    step: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentRunLogger:
    """Records a single agent run as an ordered, step-correlated trace.

    Usage: ``start_run`` once, ``set_step`` at each step boundary, then
    ``log_llm_call`` / ``log_action`` / ``log_state_change`` as events occur, and
    ``end_run`` when finished. Entries are appended immediately, so a run that
    aborts mid-flight still leaves a coherent partial trace.
    """

    def __init__(self, run_id: str, *, clock: Callable[[], str] | None = None) -> None:
        self.run_id = run_id
        self._clock = clock or _utc_now
        self._entries: list[LogEntry] = []
        self._seq = 0
        self._current_step: int | None = None
        self._status = "incomplete"
        self._error: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def start_run(self, *, metadata: dict[str, Any] | None = None) -> None:
        self._append(RUN_START, dict(metadata or {}), step=None)

    def end_run(self, *, status: str = "completed", error: str | None = None) -> None:
        self._status = status
        self._error = error
        payload: dict[str, Any] = {"status": status}
        if error is not None:
            payload["error"] = error
        self._append(RUN_END, payload, step=None)

    def set_step(self, step: int) -> None:
        self._current_step = step

    # -- event recording ---------------------------------------------------

    def log_llm_call(
        self,
        *,
        prompt: Any = None,
        tool_call: Any = None,
        response: Any = None,
        step: int | None = None,
    ) -> None:
        self._append(
            LLM_CALL,
            {"prompt": prompt, "tool_call": tool_call, "response": response},
            step=self._resolve_step(step),
        )

    def log_action(
        self,
        *,
        action: Any,
        result: Any = None,
        reward: float | None = None,
        step: int | None = None,
    ) -> None:
        self._append(
            ACTION,
            {"action": action, "result": result, "reward": reward},
            step=self._resolve_step(step),
        )

    def log_state_change(
        self, *, before: Any, after: Any, step: int | None = None
    ) -> None:
        self._append(
            STATE_CHANGE,
            {"before": before, "after": after},
            step=self._resolve_step(step),
        )

    def _resolve_step(self, step: int | None) -> int | None:
        return self._current_step if step is None else step

    # -- access ------------------------------------------------------------

    @property
    def entries(self) -> tuple[LogEntry, ...]:
        return tuple(self._entries)

    def entries_for_step(self, step: int) -> tuple[LogEntry, ...]:
        return tuple(e for e in self._entries if e.step == step)

    def trace(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self._status,
            "error": self._error,
            "entries": [e.model_dump() for e in self._entries],
        }

    def to_jsonl(self) -> str:
        return "\n".join(e.model_dump_json() for e in self._entries)

    # -- internal ----------------------------------------------------------

    def _append(self, kind: str, payload: dict[str, Any], *, step: int | None) -> None:
        # `step` is used verbatim; callers resolve the current-step default. This
        # keeps run_start/run_end unconditionally step-less even after set_step().
        entry = LogEntry(
            seq=self._seq,
            run_id=self.run_id,
            kind=kind,
            ts=self._clock(),
            step=step,
            payload=payload,
        )
        self._entries.append(entry)
        self._seq += 1


def run_logged_episode(
    env: Any,
    agent: Any,
    logger: AgentRunLogger,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
) -> AgentRunLogger:
    """Drive an agent against an env, recording a full trace into ``logger``.

    The agent logs its own LLM call (it alone holds the prompt/response); this
    driver attaches ``logger`` to the agent, then records each action, its result,
    and the resulting state change per step. On any exception the run is closed as
    ``failed`` with the error and the partial trace is preserved before re-raising.
    """
    if hasattr(agent, "logger"):
        agent.logger = logger

    logger.start_run(metadata={"seed": seed})
    try:
        obs, _info = env.reset(seed=seed)
        step = 0
        terminated = truncated = False
        while not (terminated or truncated):
            if max_steps is not None and step >= max_steps:
                break
            logger.set_step(step)
            before = obs
            action = agent.act(obs, env.action_types)
            obs, reward, terminated, truncated, info = env.step(action)
            logger.log_action(action=action, result=info, reward=reward)
            logger.log_state_change(before=before, after=obs)
            step += 1
    except Exception as exc:
        logger.end_run(status="failed", error=str(exc))
        raise
    logger.end_run(status="completed")
    return logger
