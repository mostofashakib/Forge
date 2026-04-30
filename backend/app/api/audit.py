from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import AuditLog, Episode

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/")
def list_audit_logs(
    env_name: str = Query(..., description="Filter by environment name"),
    episode_id: str | None = Query(None),
    severity: str | None = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    stmt = (
        select(AuditLog)
        .join(Episode, AuditLog.episode_id == Episode.id)
        .where(Episode.env_name == env_name)
    )
    if episode_id:
        stmt = stmt.where(AuditLog.episode_id == episode_id)
    if severity:
        stmt = stmt.where(AuditLog.severity == severity)
    stmt = stmt.order_by(AuditLog.created_at.desc()).limit(limit)
    logs = list(db.execute(stmt).scalars())
    return [
        {
            "id": log.id,
            "episode_id": log.episode_id,
            "step_index": log.step_index,
            "actor": log.actor,
            "action_type": log.action_type,
            "rule_id": log.rule_id,
            "violation": log.violation,
            "severity": log.severity,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]
