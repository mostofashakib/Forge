"""Quarantine tasks that crash on every attempt and flag suspect environment versions."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("forge.reliability")


def record_task_quarantine_if_needed(
    db_session: Any,
    env_name: str,
    task_id: str,
    environment_version: str,
    reason: str,
) -> None:
    """Quarantine task when it crashes on every allowed attempt."""
    from backend.app.models import QuarantinedTask

    try:
        existing = (
            db_session.query(QuarantinedTask)
            .filter(
                QuarantinedTask.env_name == env_name,
                QuarantinedTask.task_id == task_id,
                QuarantinedTask.released == False,
            )
            .first()
        )
        if existing is None:
            db_session.add(
                QuarantinedTask(
                    task_id=task_id,
                    env_name=env_name,
                    environment_version=environment_version,
                    reason=reason,
                    released=False,
                )
            )
            db_session.commit()
            logger.warning("[reliability] Quarantined task %s in env %s: %s", task_id, env_name, reason)
    except Exception as exc:
        logger.error("[reliability] Failed to record task quarantine: %s", exc)


def record_flagged_version_if_needed(
    db_session: Any,
    env_name: str,
    version: str,
    reason: str,
) -> None:
    """Flag an environment version for investigation."""
    from backend.app.models import FlaggedEnvironmentVersion

    try:
        existing = (
            db_session.query(FlaggedEnvironmentVersion)
            .filter(
                FlaggedEnvironmentVersion.env_name == env_name,
                FlaggedEnvironmentVersion.version == version,
                FlaggedEnvironmentVersion.status == "investigating",
            )
            .first()
        )
        if existing is None:
            db_session.add(
                FlaggedEnvironmentVersion(
                    env_name=env_name,
                    version=version,
                    reason=reason,
                    status="investigating",
                )
            )
            db_session.commit()
            logger.warning("[reliability] Flagged environment %s version %s: %s", env_name, version, reason)
    except Exception as exc:
        logger.error("[reliability] Failed to flag environment version: %s", exc)


def release_quarantined_task(db_session: Any, quarantine_id: int) -> bool:
    """Release a quarantined task."""
    from backend.app.models import QuarantinedTask

    task = db_session.get(QuarantinedTask, quarantine_id)
    if task is not None and not task.released:
        task.released = True
        task.released_at = datetime.now(timezone.utc)
        db_session.commit()
        return True
    return False


def resolve_flagged_version(db_session: Any, version_id: int) -> bool:
    """Mark a flagged environment version as resolved."""
    from backend.app.models import FlaggedEnvironmentVersion

    flagged = db_session.get(FlaggedEnvironmentVersion, version_id)
    if flagged is not None and flagged.status != "resolved":
        flagged.status = "resolved"
        flagged.resolved_at = datetime.now(timezone.utc)
        db_session.commit()
        return True
    return False
