"""Reliability API: manage quarantined tasks and flagged environment versions."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import FlaggedEnvironmentVersion, QuarantinedTask
from forge.runtime.reliability import release_quarantined_task, resolve_flagged_version

router = APIRouter(prefix="/api/reliability")


@router.get("/environments/{env_name}")
def get_environment_reliability(env_name: str, db: Session = Depends(get_db)) -> dict:
    """Return quarantined tasks and flagged versions for an environment."""
    quarantined = (
        db.query(QuarantinedTask)
        .filter(QuarantinedTask.env_name == env_name)
        .order_by(QuarantinedTask.quarantined_at.desc())
        .all()
    )
    flagged = (
        db.query(FlaggedEnvironmentVersion)
        .filter(FlaggedEnvironmentVersion.env_name == env_name)
        .order_by(FlaggedEnvironmentVersion.flagged_at.desc())
        .all()
    )
    return {
        "quarantined_tasks": [
            {
                "id": q.id,
                "task_id": q.task_id,
                "env_name": q.env_name,
                "environment_version": q.environment_version,
                "reason": q.reason,
                "released": q.released,
                "quarantined_at": q.quarantined_at.isoformat() if q.quarantined_at else None,
                "released_at": q.released_at.isoformat() if q.released_at else None,
            }
            for q in quarantined
        ],
        "flagged_versions": [
            {
                "id": f.id,
                "env_name": f.env_name,
                "version": f.version,
                "reason": f.reason,
                "status": f.status,
                "flagged_at": f.flagged_at.isoformat() if f.flagged_at else None,
                "resolved_at": f.resolved_at.isoformat() if f.resolved_at else None,
            }
            for f in flagged
        ],
    }


@router.post("/quarantine/{quarantine_id}/release")
def release_task(quarantine_id: int, db: Session = Depends(get_db)) -> dict:
    success = release_quarantined_task(db, quarantine_id)
    if not success:
        raise HTTPException(status_code=404, detail="Quarantined task not found or already released")
    return {"released": True, "id": quarantine_id}


@router.post("/flagged-versions/{version_id}/resolve")
def resolve_version(version_id: int, db: Session = Depends(get_db)) -> dict:
    success = resolve_flagged_version(db, version_id)
    if not success:
        raise HTTPException(status_code=404, detail="Flagged version not found or already resolved")
    return {"resolved": True, "id": version_id}
