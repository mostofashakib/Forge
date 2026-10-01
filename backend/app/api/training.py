"""Training API: orchestrates policy training (GRPO, DPO, SFT, PPO) from graded rollouts, synthetic data, and preference datasets."""
from __future__ import annotations

import logging
import threading
import uuid
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db, get_session_factory
from backend.app.models import TaskBatch, TrainingRun
from backend.app.services.training_data import (
    episode_counts_by_env,
    stage_available_data,
    stage_batch_by_id,
)
from backend.app.services.training_runs import execute_training_run
from forge.paths import confined_relative_path
from forge.training.checkpoint import PolicyCheckpoint
from forge.training.trainer import (
    PolicyTrainer,
    TrainingConfig,
    TrainingObjective,
)

from forge.contracts.gpu import compute_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])


class CreateTrainingRunRequest(BaseModel):
    base_model: str
    data_dir: str = Field(..., min_length=1)
    output_dir: str = Field(default="forge_policy", min_length=1)
    objective: Literal["grpo", "dpo", "sft", "ppo"] = "grpo"
    training_mode: Literal["online", "offline"] = "online"
    max_steps: int = Field(default=500, ge=1, le=10000)
    seed: int | None = None
    train_envs: list[str] | None = None
    inference_mode: Literal["auto", "local_gpu", "api_gateway"] = "auto"
    api_gateway_url: str | None = None
    gpu_precision: Literal["bf16", "fp16", "fp32", "fp8"] = "bf16"


@router.get("/data-sources")
def get_training_data_sources(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """Enumerate datasets available for training, including synthetic batches from the Data Generator."""
    sources: list[dict[str, Any]] = []

    # 1. Batches from synthetic data generator
    try:
        batches = (
            db.query(TaskBatch)
            .filter(TaskBatch.deleted_at.is_(None))
            .order_by(TaskBatch.created_at.desc())
            .limit(20)
            .all()
        )
        for b in batches:
            label = f"Generator: {b.env_name} ({b.data_type.upper()}, {b.delivered or b.requested} items, v{b.version or 1})"
            suggested = (
                "sft" if b.data_type == "sft"
                else "dpo" if b.data_type == "preference_pairs"
                else "grpo"
            )
            sources.append({
                "id": f"generator:{b.id}",
                "batch_id": b.id,
                "label": label,
                "source_type": "generator",
                "data_type": b.data_type,
                "env_name": b.env_name,
                "path": f"exports/generator/{b.id}",
                "suggested_objective": suggested,
            })
    except Exception as exc:
        logger.warning("[training] could not query task batches: %s", exc)

    # 2. Completed environment episodes
    try:
        from backend.app.services.task_factory_targets import list_targets
        episode_counts = episode_counts_by_env(db)
        for target in list_targets(db):
            count = episode_counts.get(target.name, 0)
            if count > 0:
                sources.append({
                    "id": f"env:{target.name}",
                    "label": f"Environment Rollouts: {target.name} ({count} episodes)",
                    "source_type": "environment",
                    "data_type": "rollouts",
                    "env_name": target.name,
                    "path": f"exports/{target.name}",
                    "suggested_objective": "grpo",
                })
    except Exception as exc:
        logger.warning("[training] could not query environments: %s", exc)

    # 3. Default export directory
    sources.append({
        "id": "exports",
        "label": "Default Exports Directory (exports/)",
        "source_type": "directory",
        "data_type": "all",
        "env_name": "all",
        "path": "exports",
        "suggested_objective": "grpo",
    })

    return sources


@router.get("/hardware")
def get_training_hardware() -> dict[str, Any]:
    """Inspect local GPU devices (CUDA/MPS) and configured cloud gateway for training."""
    return compute_status()


@router.post("/runs", status_code=202)
def create_training_run(
    body: CreateTrainingRunRequest, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Launch a GRPO, DPO, SFT, or PPO policy training job from exported rollouts or generator datasets."""
    root = Path.cwd()
    data_dir_param = body.data_dir.strip()

    # Handle generator batch selection (e.g. generator:tb_123 or tb_123)
    if data_dir_param.startswith("generator:") or data_dir_param.startswith("tb_") or "generator/tb_" in data_dir_param:
        batch_id = data_dir_param.split(":")[-1].split("/")[-1]
        data_dir = stage_batch_by_id(db, batch_id, root / "exports" / "generator" / batch_id)
    else:
        try:
            data_dir = confined_relative_path(root, data_dir_param)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            if not stage_available_data(db, data_dir):
                data_dir.rmdir()
                raise HTTPException(
                    status_code=422,
                    detail="No training data found. Run agent episodes or generate a task batch first.",
                )

    try:
        output_dir = confined_relative_path(root, body.output_dir)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    run_id = f"tr_{uuid.uuid4().hex[:12]}"
    run = TrainingRun(
        id=run_id,
        status="queued",
        objective=body.objective,
        training_mode=body.training_mode,
        base_model=body.base_model,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        inference_mode=body.inference_mode,
        max_steps=body.max_steps,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()

    objective_enum = TrainingObjective(body.objective.lower())
    config = TrainingConfig(
        data_dir=data_dir,
        base_model=body.base_model,
        output_dir=output_dir,
        objective=objective_enum,
        training_mode=body.training_mode,
        max_steps=body.max_steps,
        train_envs=body.train_envs,
        seed=body.seed,
        run_id=run_id,
    )

    thread = threading.Thread(
        target=partial(
            execute_training_run,
            run_id,
            config,
            session_factory=get_session_factory(),
            trainer=PolicyTrainer(),
        ),
        daemon=True,
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
                "training_mode": getattr(cp, "training_mode", "online"),
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
        "training_mode": getattr(run, "training_mode", "online"),
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
