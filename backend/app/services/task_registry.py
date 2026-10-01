"""Step 4 of the task factory: store each batch as a version.

A batch row is created when the job is queued. Saving a finished pipeline
result assigns the next version for its environment and writes the batch,
its tasks, its rejections, and its taxonomy copy in one transaction.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.models import GeneratedTask, TaskBatch, TaskRejectionRecord
from forge.taskfactory.model_settings import ModelSpec
from forge.taskfactory.pipeline import PipelineResult


def _label(spec: ModelSpec) -> str:
    return f"{spec.provider}:{spec.model}"


def create_batch(
    db: Session,
    *,
    env_name: str,
    requested: int,
    pass_k: int,
    writer: ModelSpec,
    validator: ModelSpec,
    data_type: str = "rl_tasks",
) -> str:
    batch = TaskBatch(
        id=f"tb_{uuid.uuid4().hex[:12]}",
        env_name=env_name,
        status="queued",
        requested=requested,
        pass_k=pass_k,
        data_type=data_type,
        writer_model=_label(writer),
        validator_model=_label(validator),
    )
    db.add(batch)
    db.commit()
    return batch.id


def mark_running(db: Session, batch_id: str, *, writer: ModelSpec, validator: ModelSpec) -> None:
    """Start the batch, recording the models that run it (settings may have changed since it was queued)."""
    batch = db.get(TaskBatch, batch_id)
    if batch is not None:
        batch.status = "running"
        batch.writer_model = _label(writer)
        batch.validator_model = _label(validator)
        db.commit()


def mark_failed(db: Session, batch_id: str, error: str) -> None:
    batch = db.get(TaskBatch, batch_id)
    if batch is None:
        return
    batch.status = "failed"
    batch.error = error
    batch.completed_at = datetime.now(timezone.utc)
    db.commit()


def save_result(db: Session, batch_id: str, result: PipelineResult) -> int:
    """Store a finished batch and return its version."""
    batch = db.get(TaskBatch, batch_id)
    if batch is None or batch.deleted_at is not None:
        raise ValueError(f"batch {batch_id} does not exist or was deleted")
    if batch.version is not None:
        raise ValueError(f"batch {batch_id} is already saved as version {batch.version}")
    # Deleted batches count, so a deleted version number is never reused.
    latest = db.query(func.max(TaskBatch.version)).filter(TaskBatch.env_name == batch.env_name).scalar()
    batch.version = (latest or 0) + 1
    batch.status = result.status
    batch.delivered = len(result.tasks)
    batch.data_type = getattr(result, "data_type", getattr(batch, "data_type", "rl_tasks"))
    batch.taxonomy_json = result.taxonomy.model_dump_json()
    batch.completed_at = datetime.now(timezone.utc)
    db.add_all(
        GeneratedTask(batch_id=batch_id, task_id=task.id, task_json=task.model_dump_json())
        for task in result.tasks
    )
    db.add_all(
        TaskRejectionRecord(batch_id=batch_id, rejection_json=rejection.model_dump_json())
        for rejection in result.rejections
    )
    db.commit()
    return batch.version


def delete_batch(db: Session, batch_id: str) -> bool:
    batch = db.get(TaskBatch, batch_id)
    if batch is None or batch.deleted_at is not None:
        return False
    db.query(GeneratedTask).filter(GeneratedTask.batch_id == batch_id).delete()
    db.query(TaskRejectionRecord).filter(TaskRejectionRecord.batch_id == batch_id).delete()
    batch.deleted_at = datetime.now(timezone.utc)
    db.commit()
    return True


def _summary(batch: TaskBatch) -> dict:
    return {
        "id": batch.id,
        "env_name": batch.env_name,
        "version": batch.version,
        "status": batch.status,
        "data_type": getattr(batch, "data_type", "rl_tasks"),
        "requested": batch.requested,
        "delivered": batch.delivered,
        "shortfall": batch.requested - batch.delivered if batch.version is not None else None,
        "pass_k": batch.pass_k,
        "writer_model": batch.writer_model,
        "validator_model": batch.validator_model,
        "error": batch.error,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
    }


def list_batches(db: Session, env_name: str | None = None) -> list[dict]:
    query = db.query(TaskBatch).filter(TaskBatch.deleted_at.is_(None))
    if env_name:
        query = query.filter(TaskBatch.env_name == env_name)
    return [_summary(b) for b in query.order_by(TaskBatch.created_at.desc(), TaskBatch.id.desc()).all()]


def get_batch(db: Session, batch_id: str) -> dict | None:
    batch = db.get(TaskBatch, batch_id)
    if batch is None or batch.deleted_at is not None:
        return None
    tasks = db.query(GeneratedTask).filter(GeneratedTask.batch_id == batch_id).order_by(GeneratedTask.task_id).all()
    rejections = (
        db.query(TaskRejectionRecord).filter(TaskRejectionRecord.batch_id == batch_id).order_by(TaskRejectionRecord.id).all()
    )
    return {
        **_summary(batch),
        "taxonomy": json.loads(batch.taxonomy_json) if batch.taxonomy_json else None,
        "tasks": [json.loads(t.task_json) for t in tasks],
        "rejections": [json.loads(r.rejection_json) for r in rejections],
    }
