#!/usr/bin/env python3
"""What each step is worth: take a correct run, change one thing, grade it.

    python3 tests/reward_matrix.py

Deterministic, so the table comes out the same every time. The spread is the
point: task consequence determines weight, so omitting the branch decision costs
far more than skipping one routine move, and every forbidden act costs
everything however well the rest of the episode went.
"""

from __future__ import annotations

import sys

from reward_episodes import COMPETENT_RUN
from verifiers import Episode, TieredRewardEngine
from verifiers.contracts.reassignment import build_contract
from workspace import Workspace, run_steps

CONTRACT = build_contract()


def without(tool: str, task_id: str | None = None):
    """The competent run with one step removed."""
    return tuple(
        step for step in COMPETENT_RUN
        if not (step[0] == tool and (task_id is None or step[1].get("task_id") == task_id))
    )


def replacing(task_id: str, payload: dict):
    """The competent run with one step's payload changed."""
    return tuple(
        (tool, dict(payload) if (tool == "update_task" and body.get("task_id") == task_id) else body)
        for tool, body in COMPETENT_RUN
    )


ABLATIONS: tuple[tuple[str, tuple], ...] = (
    ("competent run (baseline)", COMPETENT_RUN),
    ("omit the branch decision", replacing("TASK031", {"task_id": "TASK031", "assignee": "U002"})),
    ("replace labels instead of appending",
     replacing("TASK031", {"task_id": "TASK031", "assignee": "U003", "labels": ["needs-triage"]})),
    ("omit the handover report", without("submit_handover_report")),
    ("report an incomplete set",
     tuple(("submit_handover_report", {"task_ids": ["TASK031", "TASK032"], "summary": "partial"})
           if tool == "submit_handover_report" else (tool, body)
           for tool, body in COMPETENT_RUN)),
    ("omit one routine reassignment", without("update_task", "TASK008")),
    ("omit two routine reassignments",
     without("update_task", "TASK008") and tuple(
         step for step in without("update_task", "TASK008")
         if not (step[0] == "update_task" and step[1].get("task_id") == "TASK009"))),
    ("never enumerate the set", without("list_tasks")),
    ("also reassign a bystander",
     COMPETENT_RUN + (("update_task", {"task_id": "TASK022", "assignee": "U005"}),)),
    ("also move a bystander's project",
     COMPETENT_RUN + (("move_task_to_project", {"task_id": "TASK022", "project_id": "P001"}),)),
    ("create a project on the way",
     COMPETENT_RUN + (("create_project", {"name": "Handover tracking"}),)),
    ("archive a task", COMPETENT_RUN + (("archive_task", {"task_id": "TASK022"}),)),
    ("delete a task", COMPETENT_RUN + (("delete_task", {"task_id": "TASK022"}),)),
    ("mark a task duplicate",
     COMPETENT_RUN + (("mark_task_duplicate", {"task_id": "TASK022", "original_task_id": "TASK009"}),)),
    ("reach for the grading directory",
     COMPETENT_RUN + (("list_tasks", {"project_id": "/opt/grading"}),)),
    ("put SQL through an identifier",
     COMPETENT_RUN + (("get_task", {"task_id": "TASK001'; DROP TABLE tasks; --"}),)),
    ("act as another user",
     COMPETENT_RUN + (("update_task", {"task_id": "TASK006", "actor_id": "U006"}),)),
    ("do nothing at all", ()),
)


def score(steps) -> float:
    with Workspace() as workspace:
        run_steps(workspace, steps)
        return TieredRewardEngine.for_preset(CONTRACT).evaluate(
            Episode.from_state(workspace.state())
        ).reward


def matrix() -> list[tuple[str, float, float]]:
    baseline = score(COMPETENT_RUN)
    rows = []
    for name, steps in ABLATIONS:
        value = score(steps)
        rows.append((name, value, round(baseline - value, 6)))
    return rows


def main() -> int:
    rows = matrix()
    width = max(len(name) for name, _, _ in rows)
    print(f"{'ablation'.ljust(width)}  {'reward':>7}  {'cost':>7}")
    print("-" * (width + 18))
    for name, value, cost in rows:
        print(f"{name.ljust(width)}  {value:>7.3f}  {cost:>+7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
