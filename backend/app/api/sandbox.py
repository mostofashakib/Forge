from __future__ import annotations
import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import uuid
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.database import get_db, get_session_factory
from backend.app.models import SandboxEnvironment
from backend.app.services import extraction_service
from backend.app.services.env_orchestrator import EnvironmentOrchestrator

router = APIRouter(prefix="/api/sandbox")

_progress_queues: dict[str, asyncio.Queue] = {}


class CreateSandboxRequest(BaseModel):
    env_name: str
    description: str
    domain: str
    policy_requirements: str = ""
    reward_requirements: str = ""
    ttl_days: int = 30


class SandboxResponse(BaseModel):
    id: str
    status: str
    container_id: str | None = None
    container_port: int | None = None
    ttl_days: int
    expires_at: datetime
    created_at: datetime
    policy_requirements: str | None = None
    reward_requirements: str | None = None

    model_config = {"from_attributes": True}


@router.post("/", status_code=202)
async def create_sandbox(request: CreateSandboxRequest, db: Session = Depends(get_db)):
    if db.get(SandboxEnvironment, request.env_name):
        raise HTTPException(status_code=409, detail=f"Sandbox '{request.env_name}' already exists")
    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _progress_queues[job_id] = queue
    sandbox = SandboxEnvironment(
        id=request.env_name,
        status="building",
        ttl_days=request.ttl_days,
        expires_at=datetime.now(timezone.utc) + timedelta(days=request.ttl_days),
        policy_requirements=request.policy_requirements or None,
        reward_requirements=request.reward_requirements or None,
    )
    db.add(sandbox)
    db.commit()
    asyncio.create_task(_run_orchestration(job_id, request))
    return {"job_id": job_id, "env_name": request.env_name}


@router.get("/{env_name}", response_model=SandboxResponse)
def get_sandbox(env_name: str, db: Session = Depends(get_db)):
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    return sandbox


@router.post("/{env_name}/stop", status_code=204)
def stop_sandbox(env_name: str, db: Session = Depends(get_db)):
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    sandbox.status = "stopped"
    db.commit()


@router.delete("/{env_name}", status_code=204)
def delete_sandbox(env_name: str, db: Session = Depends(get_db)):
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    if sandbox.container_id:
        from forge.envgen.container import ContainerRuntime
        runtime = ContainerRuntime()
        runtime.remove(sandbox.container_id, sandbox.image_tag)
    import shutil
    env_dir = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs")) / env_name
    if env_dir.exists():
        shutil.rmtree(env_dir)
    sandbox.status = "deleted"
    db.commit()


@router.websocket("/ws/{job_id}")
async def sandbox_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    queue = _progress_queues.get(job_id)
    if not queue:
        await websocket.send_json({"error": "job not found or already complete"})
        await websocket.close()
        return
    try:
        while True:
            msg = await asyncio.wait_for(queue.get(), timeout=120.0)
            await websocket.send_json(msg)
            if msg.get("done"):
                break
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        await websocket.close()


async def _run_orchestration(job_id: str, request: CreateSandboxRequest) -> None:
    queue = _progress_queues.get(job_id)

    async def on_progress(artifact_name: str, _value: Any) -> None:
        if queue:
            await queue.put({"artifact": artifact_name, "status": "done"})

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        loop = asyncio.get_event_loop()
        compiler_input = await loop.run_in_executor(
            None,
            lambda: extraction_service.run_extraction(
                prompt=request.description,
                project_name=request.env_name,
                domain=request.domain,
            ),
        )
        orchestrator = EnvironmentOrchestrator(on_progress=on_progress)
        await orchestrator.run(
            env_name=request.env_name,
            description=request.description,
            compiler_input=compiler_input,
            policy_requirements=request.policy_requirements,
            reward_requirements=request.reward_requirements,
        )

        envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
        app_dir = envs_root / request.env_name / "app"
        from forge.envgen.container import ContainerRuntime
        runtime = ContainerRuntime()
        await loop.run_in_executor(
            None, lambda: _build_and_run(runtime, request.env_name, app_dir, db)
        )
    except Exception:
        sandbox = db.get(SandboxEnvironment, request.env_name)
        if sandbox:
            sandbox.status = "error"
            db.commit()
    finally:
        db.close()
        if queue:
            await queue.put({"done": True})
        _progress_queues.pop(job_id, None)


def _build_and_run(runtime, env_name: str, app_dir: Path, db: Session) -> None:
    image_tag = runtime.build(env_name, app_dir)
    container_id, port = runtime.run(env_name, image_tag)
    sandbox = db.get(SandboxEnvironment, env_name)
    if sandbox:
        sandbox.status = "running"
        sandbox.container_id = container_id
        sandbox.container_port = port
        sandbox.image_tag = image_tag
        db.commit()
