"""What a route needs before it acts, or the HTTP error that says why it cannot have it."""
from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.app.models import CalibrationRun, TaskBatch
from backend.app.services.download_gate import download_gate
from backend.app.services.export_writers.gate import DatasetGate
from forge.safety.config import SafetyConfigError


def live_batch(db: Session, batch_id: str) -> TaskBatch:
    """A batch that exists and was not deleted, else 404."""
    batch = db.get(TaskBatch, batch_id)
    if batch is None or batch.deleted_at is not None:
        raise HTTPException(status_code=404, detail="batch not found")
    return batch


def batch_run(db: Session, batch_id: str, run_id: str) -> CalibrationRun:
    """A calibration run of a live batch, else 404. A run of another batch is not found."""
    live_batch(db, batch_id)
    run = db.get(CalibrationRun, run_id)
    if run is None or run.batch_id != batch_id:
        raise HTTPException(status_code=404, detail="calibration not found")
    return run


def screening_gate(env_name: str, label: str) -> DatasetGate:
    """The safety gate for one download, else 503 naming the setting that is wrong."""
    try:
        return download_gate(env_name, label)
    except SafetyConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
