"""Shared bookkeeping for runs that fan out into episodes (agent runs, rollout jobs)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import update

from backend.app.models import AgentRun, RolloutJob


def count_finished_episode(model: type[AgentRun] | type[RolloutJob], record_id: str) -> None:
    """Atomically increment a run's episode counter; mark it completed when all are done."""
    from backend.app.database import get_session_factory

    with get_session_factory()() as db:
        db.execute(
            update(model)
            .where(model.id == record_id)
            .values(episodes_completed=model.episodes_completed + 1)
        )
        db.commit()
        record = db.get(model, record_id)
        if record and record.episodes_completed >= record.num_episodes:
            record.status = "completed"
            record.completed_at = datetime.now(timezone.utc)
            db.commit()

def mark_dispatch_failed(
    model: type[AgentRun] | type[RolloutJob], record_id: str, exc: Exception
) -> None:
    """Fail a run whose episodes could not be queued."""
    from backend.app.database import get_session_factory

    with get_session_factory()() as db:
        record = db.get(model, record_id)
        if record is not None:
            record.status = "failed"
            record.error = str(exc)
            record.completed_at = datetime.now(timezone.utc)
            db.commit()
