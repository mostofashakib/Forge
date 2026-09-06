"""What a verifier is allowed to look at.

An `Episode` is the graded record of one run: the world's end state and the
world's log of what the agent did. Both are produced by the environment, in a
container the agent has no filesystem or socket path to, which is what lets
trajectory and side-effect checks mean anything. An agent-authored trajectory
file would be evidence written by the party being graded.

There is no second input. The agent's stated answer is read here too -- but from
the `reports` table the world wrote when `submit_handover_report` ran, not from
anything in `/logs/agent`. What the agent claims and what the world recorded it
claiming are the same row, and only the world could have written it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Action:
    """One tool call, as the world recorded it."""

    seq: int
    #: When the world recorded it, in Unix milliseconds. Named `ts` because the
    #: framework's temporal checks address it by that name on every task; the
    #: unit is the tracker's, the contract is the framework's.
    ts: int
    actor_id: str
    tool: str
    task_id: str | None = None
    project_id: str | None = None
    observed_count: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Action":
        return cls(
            seq=int(row.get("seq", 0)),
            ts=int(row.get("ms") or 0),
            actor_id=str(row.get("actor_id", "")),
            tool=str(row.get("tool", "")),
            task_id=row.get("task_id"),
            project_id=row.get("project_id"),
            observed_count=int(row.get("observed_count") or 0),
        )


@dataclass(frozen=True, slots=True)
class Episode:
    state: dict[str, Any]
    actions: tuple[Action, ...] = ()
    actor_id: str = "U001"
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Whether the world recorded an action log at all. An export from before
    #: the log existed is a grading gap; an empty log is an agent that did
    #: nothing. Scoring both as zero would blame a model for a schema change.
    has_action_log: bool = True

    # -- construction ----------------------------------------------------

    @classmethod
    def from_state(cls, state: dict[str, Any], actor_id: str = "U001") -> "Episode":
        payload = state.get("result", state) if isinstance(state, dict) else {}
        actions = tuple(Action.from_row(row) for row in payload.get("action_log", []))
        return cls(
            state=payload,
            actions=actions,
            actor_id=actor_id,
            has_action_log="action_log" in payload,
        )

    @classmethod
    def from_files(cls, state_path: str | Path, actor_id: str = "U001") -> "Episode":
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        return cls.from_state(state, actor_id)

    # -- views the verifiers share ---------------------------------------

    @property
    def events(self) -> dict[str, str]:
        return {
            str(event["event_id"]): str(event["status"])
            for event in self.state.get("scenario_events", [])
        }

    @property
    def activated(self) -> set[str]:
        return {event for event, status in self.events.items() if status == "activated"}

    def activation_ts(self, event_id: str) -> str | None:
        """When an event fired, as a string so orderings compare uniformly.

        The world stores milliseconds; the framework's temporal checks compare
        with `float()`, and a numeric string converts exactly.
        """
        for event in self.state.get("scenario_events", []):
            if event["event_id"] == event_id:
                stamp = event.get("activated_ms")
                return None if stamp is None else str(stamp)
        return None

    @property
    def tasks(self) -> dict[str, dict[str, Any]]:
        return {str(task["task_id"]): task for task in self.state.get("tasks", [])}

    def task(self, task_id: str) -> dict[str, Any] | None:
        return self.tasks.get(task_id)

    def assignee(self, task_id: str) -> str | None:
        task = self.task(task_id)
        return None if task is None else task.get("assignee_id")

    def labels(self, task_id: str) -> list[str]:
        task = self.task(task_id)
        if task is None:
            return []
        raw = task.get("labels")
        if isinstance(raw, list):
            return [str(item) for item in raw]
        try:
            return list(json.loads(raw or "[]"))
        except (json.JSONDecodeError, TypeError):
            return []

    def tasks_assigned_to(self, user_id: str) -> set[str]:
        return {
            task_id for task_id, task in self.tasks.items()
            if task.get("assignee_id") == user_id
        }

    @property
    def audit_events(self) -> list[dict[str, Any]]:
        return list(self.state.get("audit_events", []))

    def audited_tasks(self) -> set[str]:
        """Every task the world recorded a mutation against.

        The audit log is how "which records did this episode touch" is answered
        without trusting either the agent's account or a diff that cannot tell a
        change from a change and back.
        """
        return {str(event["task_id"]) for event in self.audit_events}

    @property
    def reports(self) -> list[dict[str, Any]]:
        """What the acting user stated, oldest first."""
        parsed: list[dict[str, Any]] = []
        for row in self.state.get("reports", []):
            try:
                task_ids = list(json.loads(row.get("task_ids_json") or "[]"))
            except (json.JSONDecodeError, TypeError):
                task_ids = []
            parsed.append({
                "report_id": row.get("report_id"),
                "ms": row.get("ms"),
                "task_ids": [str(item) for item in task_ids],
                "summary": row.get("summary", ""),
            })
        return parsed

    @property
    def standing_report(self) -> dict[str, Any] | None:
        """The statement the episode stood behind: the last one it made.

        An episode that reported twice made two statements, and a correction is
        a correction. Grading the last one is what lets an agent fix a mistake
        without the earlier version still counting against it.
        """
        reports = self.reports
        return reports[-1] if reports else None

    @property
    def reported_task_ids(self) -> set[str]:
        report = self.standing_report
        return set(report["task_ids"]) if report else set()

    @property
    def integrity_violations(self) -> list[dict[str, Any]]:
        """Attempts to leave the environment, as the world recorded them."""
        return list(self.state.get("integrity_violations", []))

    @property
    def agent_actions(self) -> tuple[Action, ...]:
        return tuple(action for action in self.actions if action.actor_id == self.actor_id)

    def actions_with(self, tool: str) -> tuple[Action, ...]:
        return tuple(action for action in self.agent_actions if action.tool == tool)

    @property
    def tools_used(self) -> set[str]:
        return {action.tool for action in self.agent_actions}

    def namespace(self) -> dict[str, Any]:
        """The names a `PolicyVerifier` expression may use."""
        return {
            "state": self.state,
            "events": self.events,
            "activated": self.activated,
            "tasks": self.tasks,
            "reports": self.reports,
            "reported": self.reported_task_ids,
            "actions": self.agent_actions,
            "tools_used": self.tools_used,
            "actor_id": self.actor_id,
            "episode": self,
            "len": len, "any": any, "all": all, "sorted": sorted, "sum": sum,
            "set": set, "bool": bool, "int": int, "float": float, "str": str,
        }
