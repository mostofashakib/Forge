"""Fail long jobs whose worker died and left them "running"."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import or_
from sqlalchemy.orm import Session


def fail_abandoned(db: Session, model, *, older_than: timedelta, now: datetime | None, what: str) -> list[str]:
    """Fail running rows of `model` started longer ago than `older_than`, and return their ids.

    Pass the worker's hard time limit: no live worker runs a job past it, so
    a job still "running" after that lost its worker. A row from before
    `started_at` existed is judged by when it was created.
    """
    cutoff = (now or datetime.now(timezone.utc)) - older_than
    abandoned = (
        db.query(model)
        .filter(model.status == "running")
        .filter(or_(model.started_at < cutoff, (model.started_at.is_(None)) & (model.created_at < cutoff)))
        .all()
    )
    for row in abandoned:
        row.status = "failed"
        row.error = f"the worker running this {what} stopped: still running {older_than} after it started"
        row.completed_at = datetime.now(timezone.utc)
    db.commit()
    return [row.id for row in abandoned]
