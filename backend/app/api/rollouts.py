from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.services.rollout_service import create_rollout, get_rollout, list_rollouts

router = APIRouter(prefix="/api/rollouts", tags=["rollouts"])


class CreateRolloutRequest(BaseModel):
    env_name: str
    task_name: str
    agent_id: str
    num_episodes: int
    seed_start: int = 0


class RolloutJobResponse(BaseModel):
    id: str
    env_name: str
    task_name: str
    agent_id: str
    num_episodes: int
    seed_start: int
    status: str
    episodes_completed: int
    created_at: str
    completed_at: str | None = None
    error: str | None = None

    model_config = {"from_attributes": True}


@router.post("/")
def post_rollout(body: CreateRolloutRequest, db: Session = Depends(get_db)):
    job = create_rollout(
        env_name=body.env_name,
        task_name=body.task_name,
        agent_id=body.agent_id,
        num_episodes=body.num_episodes,
        seed_start=body.seed_start,
        db=db,
    )
    return {"rollout_job_id": job.id}


@router.get("/")
def list_rollout_jobs(env_name: str, db: Session = Depends(get_db)):
    jobs = list_rollouts(env_name, db)
    return [
        {
            "id": j.id,
            "env_name": j.env_name,
            "task_name": j.task_name,
            "agent_id": j.agent_id,
            "num_episodes": j.num_episodes,
            "seed_start": j.seed_start,
            "status": j.status,
            "episodes_completed": j.episodes_completed,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            "error": j.error,
        }
        for j in jobs
    ]


@router.get("/{rollout_job_id}")
def get_rollout_job(rollout_job_id: str, db: Session = Depends(get_db)):
    job = get_rollout(rollout_job_id, db)
    if job is None:
        raise HTTPException(status_code=404, detail="RolloutJob not found")
    return {
        "id": job.id,
        "env_name": job.env_name,
        "task_name": job.task_name,
        "agent_id": job.agent_id,
        "num_episodes": job.num_episodes,
        "seed_start": job.seed_start,
        "status": job.status,
        "episodes_completed": job.episodes_completed,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
    }
