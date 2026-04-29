# backend/app/api/episodes.py
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import episode_service, runner_service
from forge.runtime.replay import ReplayService

router = APIRouter(prefix="/api/episodes")


class StartEpisodeRequest(BaseModel):
    env_name: str
    task_name: str
    seed: int
    agent_id: str = "random_policy"


class StepOut(BaseModel):
    step_index: int
    action: str
    reward: float
    verifier_results: str
    diff: str
    events: str
    state_hash_before: str
    state_hash_after: str
    terminated: bool
    truncated: bool


class EpisodeOut(BaseModel):
    id: str
    env_name: str
    task_name: str
    seed: int
    agent_id: str
    status: str
    total_steps: int
    total_reward: float
    passed: bool
    steps: list[StepOut]


@router.post("/")
async def start_episode(req: StartEpisodeRequest):
    episode_id = await runner_service.start_episode(
        env_name=req.env_name,
        task_name=req.task_name,
        seed=req.seed,
        agent_id=req.agent_id,
    )
    return {"episode_id": episode_id}


@router.get("/{episode_id}", response_model=EpisodeOut)
def get_episode(episode_id: str, db: Session = Depends(get_db)):
    ep = episode_service.get_episode(episode_id, db)
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")
    steps = episode_service.get_episode_steps(episode_id, db)
    return EpisodeOut(
        id=ep.id,
        env_name=ep.env_name,
        task_name=ep.task_name,
        seed=ep.seed,
        agent_id=ep.agent_id,
        status=ep.status,
        total_steps=ep.total_steps,
        total_reward=ep.total_reward,
        passed=ep.passed,
        steps=[
            StepOut(
                step_index=s.step_index,
                action=s.action,
                reward=s.reward,
                verifier_results=s.verifier_results,
                diff=s.diff,
                events=s.events,
                state_hash_before=s.state_hash_before,
                state_hash_after=s.state_hash_after,
                terminated=s.terminated,
                truncated=s.truncated,
            )
            for s in steps
        ],
    )


@router.get("/")
def list_episodes(env_name: str, db: Session = Depends(get_db)):
    episodes = episode_service.list_episodes(env_name, db)
    return [
        {
            "id": ep.id,
            "env_name": ep.env_name,
            "task_name": ep.task_name,
            "status": ep.status,
            "passed": ep.passed,
            "total_reward": ep.total_reward,
            "total_steps": ep.total_steps,
            "started_at": ep.started_at.isoformat() if ep.started_at else None,
        }
        for ep in episodes
    ]


@router.get("/{episode_id}/steps/{step_n}/branch")
def branch(episode_id: str, step_n: int, db: Session = Depends(get_db)):
    ep = episode_service.get_episode(episode_id, db)
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")
    try:
        actions = ReplayService().branch_from(episode_id, step_n, db)
    except ValueError:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"actions": actions}
