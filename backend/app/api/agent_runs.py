from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import AgentRun, AgentEpisode, SandboxEnvironment
from backend.app.services.trajectory_files import iter_trajectory_lines, read_trajectory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sandbox", tags=["agent-runs"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class CreateAgentRunRequest(BaseModel):
    agent_id: str = "llm"
    objective: str = Field(min_length=1, max_length=20_000)
    num_episodes: int = Field(default=5, ge=1, le=1_000)
    max_steps: int = Field(default=50, ge=1, le=10_000)
    dead_end_patience: int = Field(default=5, ge=1, le=10_000)
    success_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    seed_start: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_run(env_name: str, run_id: str, db: Session) -> AgentRun:
    run = db.get(AgentRun, run_id)
    if run is None or run.env_name != env_name:
        raise HTTPException(status_code=404, detail="AgentRun not found")
    return run


def _run_to_dict(run: AgentRun) -> dict:
    return {
        "id": run.id,
        "env_name": run.env_name,
        "agent_id": run.agent_id,
        "objective": run.objective,
        "num_episodes": run.num_episodes,
        "max_steps": run.max_steps,
        "dead_end_patience": run.dead_end_patience,
        "success_threshold": run.success_threshold,
        "seed_start": run.seed_start,
        "status": run.status,
        "episodes_completed": run.episodes_completed,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


def _ep_to_dict(ep: AgentEpisode) -> dict:
    return {
        "id": ep.id,
        "run_id": ep.run_id,
        "episode_index": ep.episode_index,
        "seed": ep.seed,
        "status": ep.status,
        "total_steps": ep.total_steps,
        "total_reward": ep.total_reward,
        "final_objective_score": ep.final_objective_score,
        "termination_reason": ep.termination_reason,
        "started_at": ep.started_at.isoformat() if ep.started_at else None,
        "completed_at": ep.completed_at.isoformat() if ep.completed_at else None,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/{env_name}/agent-runs", status_code=202)
def create_agent_run(
    env_name: str,
    body: CreateAgentRunRequest,
    db: Session = Depends(get_db),
):
    sb = db.get(SandboxEnvironment, env_name)
    if sb is None:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    if sb.status != "running":
        raise HTTPException(
            status_code=409,
            detail=f"Sandbox must be running before launching agent runs (current status: {sb.status})",
        )
    if sb.container_id is None:
        raise HTTPException(status_code=409, detail="Sandbox has no active container")
    if sb.env_type == "general" and sb.container_port is None:
        raise HTTPException(status_code=409, detail="General sandbox has no container port")

    run_id = str(uuid.uuid4())
    run = AgentRun(
        id=run_id,
        env_name=env_name,
        agent_id=body.agent_id,
        objective=body.objective,
        num_episodes=body.num_episodes,
        max_steps=body.max_steps,
        dead_end_patience=body.dead_end_patience,
        success_threshold=body.success_threshold,
        seed_start=body.seed_start,
        status="pending",
        episodes_completed=0,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()

    from backend.app.worker.agent_run_tasks import run_container_run_task
    try:
        run_container_run_task.delay(run_id)
    except Exception as exc:
        logger.exception("[agent-runs] could not queue run %s", run_id)
        run.status = "failed"
        run.error = f"Could not queue the run: {exc}"
        run.completed_at = datetime.now(timezone.utc)
        db.commit()
        raise HTTPException(status_code=503, detail="Worker unavailable — could not queue the run") from exc

    return {"run_id": run_id, **_run_to_dict(run)}


@router.get("/{env_name}/agent-runs")
def list_agent_runs(env_name: str, db: Session = Depends(get_db)):
    runs = (
        db.query(AgentRun)
        .filter(AgentRun.env_name == env_name)
        .order_by(AgentRun.created_at.desc())
        .all()
    )
    return [_run_to_dict(r) for r in runs]


@router.get("/{env_name}/agent-runs/{run_id}")
def get_agent_run(env_name: str, run_id: str, db: Session = Depends(get_db)):
    run = _require_run(env_name, run_id, db)
    return _run_to_dict(run)


@router.get("/{env_name}/agent-runs/{run_id}/episodes")
def list_agent_episodes(env_name: str, run_id: str, db: Session = Depends(get_db)):
    _require_run(env_name, run_id, db)
    episodes = (
        db.query(AgentEpisode)
        .filter(AgentEpisode.run_id == run_id)
        .order_by(AgentEpisode.episode_index)
        .all()
    )
    return [_ep_to_dict(ep) for ep in episodes]


@router.get("/{env_name}/agent-runs/{run_id}/episodes/{episode_id}/trajectory")
def get_trajectory(env_name: str, run_id: str, episode_id: str, db: Session = Depends(get_db)):
    _require_run(env_name, run_id, db)
    ep = db.get(AgentEpisode, episode_id)
    if ep is None or ep.run_id != run_id:
        raise HTTPException(status_code=404, detail="Episode not found")

    if ep.jsonl_path is None:
        raise HTTPException(status_code=404, detail="Trajectory file not yet available")

    try:
        trajectory = read_trajectory(Path(ep.jsonl_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Trajectory file not yet available") from exc
    except (OSError, ValueError) as exc:
        logger.exception("Could not read trajectory %s for episode %s", ep.jsonl_path, episode_id)
        raise HTTPException(status_code=500, detail="Trajectory file is unreadable") from exc

    return {
        "episode_id": episode_id,
        "steps": trajectory.steps,
        "summary": trajectory.summary,
    }


@router.get("/{env_name}/agent-runs/{run_id}/export")
def export_trajectories(env_name: str, run_id: str, db: Session = Depends(get_db)):
    """Return all completed episode trajectories concatenated as JSONL text."""
    _require_run(env_name, run_id, db)

    paths = (
        db.query(AgentEpisode.jsonl_path)
        .filter(AgentEpisode.run_id == run_id, AgentEpisode.status == "completed")
        .order_by(AgentEpisode.episode_index)
        .all()
    )

    return StreamingResponse(
        iter_trajectory_lines(Path(path) for (path,) in paths if path),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id[:8]}_trajectories.jsonl"'},
    )


@router.delete("/{env_name}/agent-runs/{run_id}", status_code=204)
def delete_agent_run(env_name: str, run_id: str, db: Session = Depends(get_db)):
    """Delete a run, all its episodes, and their JSONL trajectory files."""
    run = _require_run(env_name, run_id, db)

    episodes = (
        db.query(AgentEpisode)
        .filter(AgentEpisode.run_id == run_id)
        .all()
    )
    for ep in episodes:
        if ep.jsonl_path:
            try:
                Path(ep.jsonl_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("[agent-runs] could not remove %s: %s", ep.jsonl_path, exc)
        db.delete(ep)

    db.delete(run)
    db.commit()
