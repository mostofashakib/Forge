"""The one definition of an active sandbox, which the environment cap counts."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.app.models import SandboxEnvironment


def active_criteria() -> list:
    """Predicate for a genuinely active sandbox.

    A sandbox is inactive if it was deleted/expired *or* its TTL has already
    lapsed — even when the periodic cleanup sweep has not yet flipped its
    status, a time-expired environment must never appear in active inventory.
    """
    return [
        SandboxEnvironment.status.notin_(["deleted", "expired"]),
        SandboxEnvironment.expires_at > datetime.now(timezone.utc),
    ]


def active_sandbox_count(db: Session) -> int:
    return db.query(SandboxEnvironment).filter(*active_criteria()).count()
