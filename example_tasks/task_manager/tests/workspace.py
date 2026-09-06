"""Shared test harness: a throwaway world driven through `execute_tool`.

Every suite needs the same three things -- a fresh database, a way to run tool
calls against it, and its state export. Doing that through `execute_tool` rather
than by writing rows keeps the tests honest: they exercise the same path the
socket server and the RL environment take, so a suite cannot pass against a
world the agent could never produce.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from task_sim.scenario import Scenario
from task_sim.seed import REASSIGNMENT_SCENARIO
from task_sim.service import execute_tool, export_state, seed_database
from task_sim.sqlite_common import ToolError, WorldError


class Workspace:
    """One disposable world. Use as a context manager, or call `close()`."""

    def __init__(self, scenario: Scenario = REASSIGNMENT_SCENARIO) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="tracker-test-"))
        self.db_path = self.root / "tasks.db"
        self.snapshot_path = self.root / "seed.sql"
        self.scenario = scenario
        seed_database(self.db_path, self.snapshot_path, scenario)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    # -- driving ---------------------------------------------------------

    def call(self, tool: str, **payload: Any) -> dict[str, Any]:
        """Run one tool. Raises whatever the world raises."""
        return execute_tool(self.db_path, tool, payload)

    def try_call(self, tool: str, **payload: Any) -> dict[str, Any] | WorldError:
        """Run one tool, returning the refusal instead of raising it."""
        try:
            return execute_tool(self.db_path, tool, payload)
        except WorldError as exc:
            return exc

    def refusal(self, tool: str, **payload: Any) -> ToolError:
        """Run one tool that must be refused, and hand back the refusal."""
        outcome = self.try_call(tool, **payload)
        if not isinstance(outcome, WorldError):
            raise AssertionError(f"{tool} was expected to be refused, but returned {outcome!r}")
        return outcome  # type: ignore[return-value]

    # -- reading ---------------------------------------------------------

    def state(self) -> dict[str, Any]:
        return export_state(self.db_path)

    def events(self) -> dict[str, str]:
        return {
            event["event_id"]: event["status"]
            for event in self.state()["scenario_events"]
        }

    def activated(self) -> set[str]:
        return {event for event, status in self.events().items() if status == "activated"}

    def task(self, task_id: str) -> dict[str, Any] | None:
        for task in self.state()["tasks"]:
            if task["task_id"] == task_id:
                return task
        return None

    def labels(self, task_id: str) -> list[str]:
        task = self.task(task_id)
        if task is None:
            return []
        raw = task["labels"]
        return list(raw) if isinstance(raw, list) else json.loads(raw or "[]")

    def visible_task_ids(self) -> set[str]:
        return {task["task_id"] for task in self.state()["tasks"]}


# ---------------------------------------------------------------------------
# The oracle, as data
# ---------------------------------------------------------------------------
#
# Several suites need a correct episode, and each writing its own would let them
# drift into testing slightly different tasks.

ORACLE_STEPS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("list_tasks", {"assignee": "U004"}),
    ("get_task", {"task_id": "TASK006"}),
    ("update_task", {"task_id": "TASK006", "assignee": "U002"}),
    ("get_task", {"task_id": "TASK008"}),
    ("update_task", {"task_id": "TASK008", "assignee": "U002"}),
    ("get_task", {"task_id": "TASK009"}),
    ("update_task", {"task_id": "TASK009", "assignee": "U002"}),
    ("get_task", {"task_id": "TASK032"}),
    ("update_task", {"task_id": "TASK032", "assignee": "U002"}),
    ("get_task", {"task_id": "TASK031"}),
    ("update_task", {
        "task_id": "TASK031", "assignee": "U003",
        "labels": ["design", "frontend", "needs-triage"],
    }),
    ("submit_handover_report", {
        "task_ids": ["TASK006", "TASK008", "TASK009", "TASK031", "TASK032"],
        "summary": "Reassigned Jordan Kim's five tasks per the milestone branch.",
    }),
)


def run_steps(workspace: Workspace, steps: Any) -> None:
    """Apply a sequence of (tool, payload) pairs, tolerating refusals.

    A refused call never happened, so an episode that tries something the world
    declines is graded on what it actually did -- which is what makes a refusal
    different from a mistake.
    """
    for tool, payload in steps:
        try:
            execute_tool(workspace.db_path, tool, dict(payload))
        except WorldError:
            pass


def oracle_state(scenario: Scenario = REASSIGNMENT_SCENARIO) -> dict[str, Any]:
    """The state export of a correct episode."""
    with Workspace(scenario) as workspace:
        run_steps(workspace, ORACLE_STEPS)
        return workspace.state()


__all__ = ["ORACLE_STEPS", "Workspace", "oracle_state", "run_steps"]
