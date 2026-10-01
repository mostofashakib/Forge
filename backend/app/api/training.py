"""Training API: orchestrates GRPO and DPO policy training from graded rollouts and preference datasets."""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db, get_session_factory
from backend.app.models import TrainingRun
from forge.paths import confined_relative_path
from forge.training.checkpoint import PolicyCheckpoint
from forge.training.trainer import (
    NoTrainingSignalError,
    PolicyTrainer,
    TrainingConfig,
    TrainingObjective,
)

from forge.contracts.gpu import APIGatewaySpec, GPUDeviceSpec, GPUInferenceContract, InferenceMode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])


class CreateTrainingRunRequest(BaseModel):
    base_model: str
    data_dir: str = Field(..., min_length=1)
    output_dir: str = Field(default="forge_policy", min_length=1)
    objective: Literal["grpo", "dpo"] = "grpo"
    max_steps: int = Field(default=500, ge=1, le=10000)
    seed: int | None = None
    train_envs: list[str] | None = None
    inference_mode: Literal["auto", "local_gpu", "api_gateway"] = "auto"
    api_gateway_url: str | None = None
    gpu_precision: Literal["bf16", "fp16", "fp32", "fp8"] = "bf16"


@router.get("/hardware")
def get_training_hardware() -> dict[str, Any]:
    """Inspect local GPU devices (CUDA/MPS) and configured cloud gateway for training."""
    hardware = GPUDeviceSpec.probe_hardware()
    gw_spec = APIGatewaySpec()
    return {
        "hardware": hardware,
        "default_mode": "local_gpu" if hardware.get("cuda_available") or hardware.get("mps_available") else "api_gateway",
        "api_gateway": {
            "endpoint_url": gw_spec.endpoint_url,
            "has_api_key": bool(gw_spec.resolved_api_key()),
        },
    }


def _execute_training_run(run_id: str, config: TrainingConfig) -> None:
    """Execute policy training in a background worker thread."""
    SessionLocal = get_session_factory()
    with SessionLocal() as db:
        run = db.get(TrainingRun, run_id)
        if not run:
            return
        run.status = "running"
        db.commit()

    try:
        trainer = PolicyTrainer()
        result = trainer.train(config)
        with SessionLocal() as db:
            run = db.get(TrainingRun, run_id)
            if run:
                run.status = "completed"
                run.checkpoint_path = result.checkpoint_path
                run.num_examples = result.num_examples
                run.mean_reward = getattr(result, "mean_reward", 0.0)
                run.completed_at = datetime.now(timezone.utc)
                db.commit()
        logger.info("[training] run %s completed: checkpoint at %s", run_id, result.checkpoint_path)
    except NoTrainingSignalError as exc:
        logger.warning("[training] run %s failed with no training signal: %s", run_id, exc)
        with SessionLocal() as db:
            run = db.get(TrainingRun, run_id)
            if run:
                run.status = "failed"
                run.error = f"No training signal: {exc}"
                run.completed_at = datetime.now(timezone.utc)
                db.commit()
    except Exception as exc:
        logger.exception("[training] run %s encountered an error: %s", run_id, exc)
        with SessionLocal() as db:
            run = db.get(TrainingRun, run_id)
            if run:
                run.status = "failed"
                run.error = str(exc)
                run.completed_at = datetime.now(timezone.utc)
                db.commit()


@router.post("/runs", status_code=202)
def create_training_run(
    body: CreateTrainingRunRequest, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Launch a GRPO or DPO policy training job from exported rollouts or preference datasets."""
    root = Path.cwd()
    try:
        data_dir = confined_relative_path(root, body.data_dir)
        output_dir = confined_relative_path(root, body.output_dir)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not data_dir.exists():
        raise HTTPException(status_code=422, detail=f"data_dir does not exist: {body.data_dir}")

    run_id = f"tr_{uuid.uuid4().hex[:12]}"
    run = TrainingRun(
        id=run_id,
        status="queued",
        objective=body.objective,
        base_model=body.base_model,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        inference_mode=body.inference_mode,
        max_steps=body.max_steps,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()

    config = TrainingConfig(
        data_dir=data_dir,
        base_model=body.base_model,
        output_dir=output_dir,
        objective=TrainingObjective.GRPO if body.objective == "grpo" else TrainingObjective.DPO,
        max_steps=body.max_steps,
        train_envs=body.train_envs,
        seed=body.seed,
        run_id=run_id,
    )

    thread = threading.Thread(
        target=_execute_training_run, args=(run_id, config), daemon=True
    )
    thread.start()

    logger.info("[training] queued run %s (objective=%s, model=%s)", run_id, body.objective, body.base_model)
    return {"run_id": run_id, "status": "queued"}


@router.get("/runs")
def list_training_runs(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """List all training runs in chronological order."""
    runs = db.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()
    return [_run_to_dict(r) for r in runs]


@router.get("/runs/{run_id}")
def get_training_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get status, loss, and checkpoint location of a specific training run."""
    run = db.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found")
    return _run_to_dict(run)


@router.get("/checkpoints")
def list_checkpoints(output_dir: str = "forge_policy") -> list[dict[str, Any]]:
    """List saved policy checkpoints found in the filesystem."""
    root = Path.cwd()
    try:
        resolved_dir = confined_relative_path(root, output_dir)
    except ValueError:
        return []

    checkpoints: list[dict[str, Any]] = []
    if not resolved_dir.exists():
        return []

    # Check root dir and immediate subdirectories
    dirs_to_check = [resolved_dir] + [p for p in resolved_dir.iterdir() if p.is_dir()]
    for d in dirs_to_check:
        try:
            cp = PolicyCheckpoint.load(d)
            checkpoints.append({
                "directory": str(d),
                "objective": cp.objective,
                "base_model": cp.base_model,
                "num_examples": cp.num_examples,
                "mean_reward": cp.mean_reward,
                "created_at": cp.created_at,
                "run_id": cp.run_id,
            })
        except Exception:
            continue

    return checkpoints


def _run_to_dict(run: TrainingRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "status": run.status,
        "objective": run.objective,
        "base_model": run.base_model,
        "data_dir": run.data_dir,
        "output_dir": run.output_dir,
        "checkpoint_path": run.checkpoint_path,
        "inference_mode": getattr(run, "inference_mode", "auto"),
        "max_steps": run.max_steps,
        "num_examples": run.num_examples,
        "mean_reward": run.mean_reward,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }
