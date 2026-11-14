"""Calibration API: measure how often a target model solves each task in a saved batch."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, WebSocket
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.api._guards import batch_run, live_batch
from backend.app.api._pubsub_relay import stream_channel
from backend.app.database import get_db, get_session_factory
from backend.app.models import CalibrationRun
from backend.app.services import calibration_registry, task_judgments, task_registry
from backend.app.worker.calibration import channel_for, run_calibration_task
from forge.settings import redis_url
from forge.taskfactory.calibration import MAX_ATTEMPTS, MIN_ATTEMPTS
from forge.taskfactory.policy import target_problem

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/task-factory")

_FINISHED = {"complete", "failed"}


class CreateCalibrationRequest(BaseModel):
    # An agent id: ollama:<model>, vllm:<model>, openai:<model>, or anthropic:<model>.
    target_model: str
    attempts: int = Field(default=MAX_ATTEMPTS, ge=MIN_ATTEMPTS, le=MAX_ATTEMPTS)


@router.post("/batches/{batch_id}/calibrations", status_code=202)
def create_calibration(batch_id: str, body: CreateCalibrationRequest, db: Session = Depends(get_db)) -> dict:
    batch = live_batch(db, batch_id)
    if batch.version is None:
        raise HTTPException(status_code=409, detail="only a saved batch can be calibrated")
    if not task_registry.batch_tasks(db, batch_id):
        raise HTTPException(status_code=409, detail="this batch has no tasks to calibrate")
    problem = target_problem(body.target_model)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    run_id = calibration_registry.create_run(
        db, batch_id=batch_id, target_model=body.target_model, attempts=body.attempts,
    )
    run_calibration_task.delay(run_id=run_id)
    logger.info("[calibration] queued %s for %s: %s, %d attempts", run_id, batch_id, body.target_model, body.attempts)
    return {"run_id": run_id}


@router.get("/batches/{batch_id}/calibrations")
def list_calibrations(batch_id: str, db: Session = Depends(get_db)) -> list[dict]:
    live_batch(db, batch_id)
    return calibration_registry.list_runs(db, batch_id)


@router.get("/batches/{batch_id}/calibrations/{run_id}")
def get_calibration(batch_id: str, run_id: str, db: Session = Depends(get_db)) -> dict:
    batch_run(db, batch_id, run_id)
    detail = calibration_registry.get_run(db, run_id)
    verdicts = task_judgments.judgments(db, batch_id)
    detail["tasks"] = [{**task, "judgment": verdicts.get(task["task_id"])} for task in detail["tasks"]]
    return detail


@router.get("/batches/{batch_id}/calibrations/{run_id}/tasks/{task_id}")
def get_task_attempts(batch_id: str, run_id: str, task_id: str, db: Session = Depends(get_db)) -> list[dict]:
    batch_run(db, batch_id, run_id)
    return calibration_registry.task_attempts(db, run_id, task_id)


def _finished_message(run_id: str) -> dict | None:
    with get_session_factory()() as db:
        run = db.get(CalibrationRun, run_id)
        if run is None:
            return {"error": "calibration not found"}
        if run.status == "failed":
            return {"error": run.error or "calibration failed"}
        if run.status in _FINISHED:
            return {"done": True}
        return None


@router.websocket("/ws/calibrations/{run_id}")
async def calibration_progress_ws(websocket: WebSocket, run_id: str) -> None:
    await stream_channel(
        websocket,
        redis_url=redis_url(),
        channel=channel_for(run_id),
        finished_message=lambda: _finished_message(run_id),
        is_final=lambda data: bool(data.get("done") or data.get("error")),
        log_tag="calibration",
    )
