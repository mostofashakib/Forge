from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import AgentRun, AgentEpisode, SandboxEnvironment

router = APIRouter(prefix="/api/sandbox", tags=["agent-runs"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class CreateAgentRunRequest(BaseModel):
    agent_id: str = "llm"
    objective: str
    num_episodes: int = 5
    max_steps: int = 50
    divergence_threshold: float = 0.2
    consecutive_below_threshold: int = 3
    dead_end_patience: int = 5
    success_threshold: float = 0.9
    seed_start: int = 0


class AgentRunResponse(BaseModel):
    id: str
    env_name: str
    agent_id: str
    objective: str
    num_episodes: int
    max_steps: int
    divergence_threshold: float
    consecutive_below_threshold: int
    dead_end_patience: int
    success_threshold: float
    seed_start: int
    status: str
    episodes_completed: int
    error: str | None = None
    created_at: str
    completed_at: str | None = None

    model_config = {"from_attributes": True}


class AgentEpisodeResponse(BaseModel):
    id: str
    run_id: str
    episode_index: int
    seed: int
    status: str
    total_steps: int
    total_reward: float
    final_objective_score: float
    termination_reason: str | None = None
    started_at: str
    completed_at: str | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_to_dict(run: AgentRun) -> dict:
    return {
        "id": run.id,
        "env_name": run.env_name,
        "agent_id": run.agent_id,
        "objective": run.objective,
        "num_episodes": run.num_episodes,
        "max_steps": run.max_steps,
        "divergence_threshold": run.divergence_threshold,
        "consecutive_below_threshold": run.consecutive_below_threshold,
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
        divergence_threshold=body.divergence_threshold,
        consecutive_below_threshold=body.consecutive_below_threshold,
        dead_end_patience=body.dead_end_patience,
        success_threshold=body.success_threshold,
        seed_start=body.seed_start,
        status="pending",
        episodes_completed=0,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()

    from backend.app.worker.tasks import run_container_run_task
    run_container_run_task.delay(run_id)

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
    run = db.get(AgentRun, run_id)
    if run is None or run.env_name != env_name:
        raise HTTPException(status_code=404, detail="AgentRun not found")
    return _run_to_dict(run)


@router.get("/{env_name}/agent-runs/{run_id}/episodes")
def list_agent_episodes(env_name: str, run_id: str, db: Session = Depends(get_db)):
    run = db.get(AgentRun, run_id)
    if run is None or run.env_name != env_name:
        raise HTTPException(status_code=404, detail="AgentRun not found")
    episodes = (
        db.query(AgentEpisode)
        .filter(AgentEpisode.run_id == run_id)
        .order_by(AgentEpisode.episode_index)
        .all()
    )
    return [_ep_to_dict(ep) for ep in episodes]


@router.get("/{env_name}/agent-runs/{run_id}/episodes/{episode_id}/trajectory")
def get_trajectory(env_name: str, run_id: str, episode_id: str, db: Session = Depends(get_db)):
    ep = db.get(AgentEpisode, episode_id)
    if ep is None or ep.run_id != run_id:
        raise HTTPException(status_code=404, detail="Episode not found")

    if ep.jsonl_path is None or not Path(ep.jsonl_path).exists():
        raise HTTPException(status_code=404, detail="Trajectory file not yet available")

    lines = Path(ep.jsonl_path).read_text(encoding="utf-8").strip().splitlines()
    steps = []
    summary = None
    for line in lines:
        record = json.loads(line)
        if record.get("type") == "episode_summary":
            summary = record
        else:
            steps.append(record)

    return {
        "episode_id": episode_id,
        "steps": steps,
        "summary": summary,
    }


@router.get("/{env_name}/agent-runs/{run_id}/export")
def export_trajectories(env_name: str, run_id: str, db: Session = Depends(get_db)):
    """Return all completed episode trajectories concatenated as JSONL text."""
    run = db.get(AgentRun, run_id)
    if run is None or run.env_name != env_name:
        raise HTTPException(status_code=404, detail="AgentRun not found")

    episodes = (
        db.query(AgentEpisode)
        .filter(AgentEpisode.run_id == run_id, AgentEpisode.status == "completed")
        .order_by(AgentEpisode.episode_index)
        .all()
    )

    all_lines: list[str] = []
    for ep in episodes:
        if ep.jsonl_path and Path(ep.jsonl_path).exists():
            all_lines.append(Path(ep.jsonl_path).read_text(encoding="utf-8").strip())

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        "\n".join(all_lines),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id[:8]}_trajectories.jsonl"'},
    )


@router.delete("/{env_name}/agent-runs/{run_id}", status_code=204)
def delete_agent_run(env_name: str, run_id: str, db: Session = Depends(get_db)):
    """Delete a run, all its episodes, and their JSONL trajectory files."""
    run = db.get(AgentRun, run_id)
    if run is None or run.env_name != env_name:
        raise HTTPException(status_code=404, detail="AgentRun not found")

    episodes = (
        db.query(AgentEpisode)
        .filter(AgentEpisode.run_id == run_id)
        .all()
    )
    for ep in episodes:
        if ep.jsonl_path:
            try:
                Path(ep.jsonl_path).unlink(missing_ok=True)
            except Exception:
                pass
        db.delete(ep)

    db.delete(run)
    db.commit()
