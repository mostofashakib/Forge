"""Calibration runs: a target model's attempts at a saved batch, one row per attempt.

Rates, bands, and failure summaries are rebuilt from the attempt rows on
read, so a failed or still-running run shows exactly what it recorded.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from backend.app.models import CalibrationAttempt, CalibrationRun
from backend.app.services import task_registry
from backend.app.services.abandoned import fail_abandoned
from forge.taskfactory.attempt import Attempt, attempt_labels, summarize_failures
from forge.taskfactory.calibration import TaskRate, calibration_summary
from forge.taskfactory.schemas import SyntheticTask


def create_run(db: Session, *, batch_id: str, target_model: str, attempts: int) -> str:
    run = CalibrationRun(
        id=f"cr_{uuid.uuid4().hex[:12]}", batch_id=batch_id, target_model=target_model,
        attempts_per_task=attempts, status="queued",
    )
    db.add(run)
    db.commit()
    return run.id


def mark_running(db: Session, run_id: str) -> None:
    run = db.get(CalibrationRun, run_id)
    if run is not None:
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        db.commit()


def complete_run(db: Session, run_id: str) -> None:
    _finish(db, run_id, "complete", None)


def mark_failed(db: Session, run_id: str, error: str) -> None:
    _finish(db, run_id, "failed", error)


def fail_abandoned_runs(db: Session, *, older_than: timedelta, now: datetime | None = None) -> list[str]:
    """Fail running calibrations whose worker died. See `fail_abandoned`."""
    return fail_abandoned(db, CalibrationRun, older_than=older_than, now=now, what="calibration")


def record_attempts(db: Session, run_id: str, task: SyntheticTask, attempts: Sequence[Attempt]) -> None:
    """Store one task's attempts, committed together."""
    db.add_all(
        CalibrationAttempt(
            run_id=run_id, task_id=task.id, attempt_index=index, passed=attempt.passed,
            termination=attempt.termination, error=attempt.error,
            steps_json=json.dumps([_turn(turn) for turn in attempt.turns]),
            checks_json=json.dumps([asdict(outcome) for outcome in attempt.checks]),
            labels_json=json.dumps(attempt_labels(attempt, task)),
        )
        for index, attempt in enumerate(attempts)
    )
    db.commit()


def list_runs(db: Session, batch_id: str) -> list[dict]:
    runs = (
        db.query(CalibrationRun).filter(CalibrationRun.batch_id == batch_id)
        .order_by(CalibrationRun.created_at.desc(), CalibrationRun.id.desc()).all()
    )
    return [_run_summary(run) for run in runs]


def get_run(db: Session, run_id: str) -> dict | None:
    run = db.get(CalibrationRun, run_id)
    if run is None:
        return None
    rows = _rows(db, run_id)
    by_task: dict[str, list[CalibrationAttempt]] = {}
    for row in rows:
        by_task.setdefault(row.task_id, []).append(row)
    tasks = {task.id: task for task in task_registry.batch_tasks(db, run.batch_id)}
    results = [_task_result(tasks[task_id], attempts) for task_id, attempts in by_task.items() if task_id in tasks]
    rates = [
        TaskRate(task_id=r["task_id"], category=r["category"], difficulty=r["difficulty"],
                 passed=r["passed"], attempts=r["attempts"], suspect=_suspect(r["failure_summary"]))
        for r in results
    ]
    return {**_run_summary(run), "tasks": results, "summary": calibration_summary(rates)}


def task_attempts(db: Session, run_id: str, task_id: str) -> list[dict]:
    return [_attempt(row) for row in _rows(db, run_id) if row.task_id == task_id]


def attempts_by_task(db: Session, run_id: str) -> dict[str, list[dict]]:
    """Every attempt of a run, grouped by task, each in the `task_attempts` shape."""
    grouped: dict[str, list[dict]] = {}
    for row in _rows(db, run_id):
        grouped.setdefault(row.task_id, []).append(_attempt(row))
    return grouped


def _attempt(row: CalibrationAttempt) -> dict:
    return {
        "attempt_index": row.attempt_index, "passed": row.passed, "termination": row.termination,
        "error": row.error, "steps": json.loads(row.steps_json), "checks": json.loads(row.checks_json),
        "labels": json.loads(row.labels_json),
    }


def suspect_tasks(db: Session, batch_id: str) -> dict[str, tuple[str, list[str]]]:
    """Tasks any calibration of the batch flagged suspect, with the run and the checks every attempt missed."""
    runs = db.query(CalibrationRun).filter(CalibrationRun.batch_id == batch_id).order_by(CalibrationRun.created_at)
    flagged: dict[str, tuple[str, list[str]]] = {}
    for run in runs:
        by_task: dict[str, list[dict]] = {}
        for row in _rows(db, run.id):
            by_task.setdefault(row.task_id, []).append(json.loads(row.labels_json))
        for task_id, labels in by_task.items():
            summary = summarize_failures(labels)
            if task_id not in flagged and summary and summary["suspect_checks"]:
                flagged[task_id] = (run.id, summary["suspect_checks"])
    return flagged


def _rows(db: Session, run_id: str) -> list[CalibrationAttempt]:
    return (
        db.query(CalibrationAttempt).filter(CalibrationAttempt.run_id == run_id)
        .order_by(CalibrationAttempt.task_id, CalibrationAttempt.attempt_index).all()
    )


def _suspect(failure_summary: dict | None) -> bool:
    return bool(failure_summary and failure_summary["suspect_checks"])


def _task_result(task: SyntheticTask, rows: list[CalibrationAttempt]) -> dict:
    passed = sum(row.passed for row in rows)
    failure_summary = summarize_failures([json.loads(row.labels_json) for row in rows])
    rate = TaskRate(task_id=task.id, category=task.category, difficulty=task.difficulty,
                    passed=passed, attempts=len(rows), suspect=_suspect(failure_summary))
    return {
        "task_id": task.id, "title": task.title, "category": task.category, "difficulty": task.difficulty,
        "edge_case": task.edge_case, "passed": passed, "attempts": len(rows),
        "solve_rate": rate.solve_rate, "band": rate.band,
        "failure_summary": failure_summary,
    }


def _turn(turn) -> dict:
    return {"step": turn.step.model_dump(), "ok": turn.result.ok, "output": turn.result.output, "error": turn.result.error}


def _finish(db: Session, run_id: str, status: str, error: str | None) -> None:
    run = db.get(CalibrationRun, run_id)
    if run is None:
        return
    run.status = status
    run.error = error
    run.completed_at = datetime.now(timezone.utc)
    db.commit()


def _run_summary(run: CalibrationRun) -> dict:
    return {
        "id": run.id, "batch_id": run.batch_id, "target_model": run.target_model,
        "attempts_per_task": run.attempts_per_task, "status": run.status, "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }
