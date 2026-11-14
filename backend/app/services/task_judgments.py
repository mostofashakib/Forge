"""A person's judgment on a saved task, and the tasks exports must leave out.

Calibration flags a task suspect when every attempt finished and missed the
same checks. That usually means the checks test something the objective never
asks for, but only a person can tell a defect from a hard task. A suspect task
stays out of exports until a person marks it hard. A task marked unfair stays
out for good.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.app.models import TaskJudgment
from backend.app.services import calibration_registry, task_registry

VERDICTS = ("hard", "unfair")


def judge(db: Session, batch_id: str, task_id: str, verdict: str) -> None:
    """Record a verdict, replacing any earlier one. Raises LookupError for a task outside the batch."""
    if verdict not in VERDICTS:
        raise ValueError(f"a judgment is hard or unfair, got {verdict!r}")
    if task_id not in {task.id for task in task_registry.batch_tasks(db, batch_id)}:
        raise LookupError(f"batch {batch_id} has no task {task_id}")
    row = db.query(TaskJudgment).filter_by(batch_id=batch_id, task_id=task_id).one_or_none()
    if row is None:
        db.add(TaskJudgment(batch_id=batch_id, task_id=task_id, verdict=verdict))
    else:
        row.verdict = verdict
        row.judged_at = datetime.now(timezone.utc)
    db.commit()


def judgments(db: Session, batch_id: str) -> dict[str, str]:
    return {row.task_id: row.verdict for row in db.query(TaskJudgment).filter_by(batch_id=batch_id)}


def held_out(db: Session, batch_id: str) -> dict[str, str]:
    """Tasks exports must leave out, each with the reason."""
    verdicts = judgments(db, batch_id)
    held = {task_id: "a person marked it unfair" for task_id, verdict in verdicts.items() if verdict == "unfair"}
    for task_id, (run_id, checks) in calibration_registry.suspect_tasks(db, batch_id).items():
        if task_id not in verdicts:
            held[task_id] = (
                f"calibration {run_id} flagged it suspect: every attempt missed {', '.join(checks)}. "
                "Mark it hard to export it."
            )
    return held
