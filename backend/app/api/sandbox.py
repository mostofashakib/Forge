from __future__ import annotations
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid
import redis
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models import SandboxEnvironment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sandbox")


class CreateSandboxRequest(BaseModel):
    env_name: str
    description: str
    domain: str = "localhost"
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
    logger.info("[sandbox] POST /api/sandbox/ — env_name=%s", request.env_name)

    existing = db.get(SandboxEnvironment, request.env_name)
    if existing:
        if existing.status in ("deleted", "expired"):
            logger.info("[sandbox] removing stale record for %s (status=%s)", request.env_name, existing.status)
            db.delete(existing)
            db.commit()
        else:
            logger.warning("[sandbox] conflict: %s already exists (status=%s)", request.env_name, existing.status)
            raise HTTPException(status_code=409, detail=f"Sandbox '{request.env_name}' already exists")

    job_id = str(uuid.uuid4())
    sandbox = SandboxEnvironment(
        id=request.env_name,
        status="queued",
        ttl_days=request.ttl_days,
        expires_at=datetime.now(timezone.utc) + timedelta(days=request.ttl_days),
        policy_requirements=request.policy_requirements or None,
        reward_requirements=request.reward_requirements or None,
    )
    db.add(sandbox)
    db.commit()
    logger.info("[sandbox] DB row created for %s (job_id=%s)", request.env_name, job_id)

    import asyncio
    from urllib.parse import urlparse
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    _parsed = urlparse(redis_url)
    _host, _port = _parsed.hostname or "localhost", _parsed.port or 6379
    logger.info("[sandbox] checking Redis reachability at %s:%s…", _host, _port)
    try:
        _reader, _writer = await asyncio.wait_for(
            asyncio.open_connection(_host, _port), timeout=3
        )
        _writer.close()
        await _writer.wait_closed()
        logger.info("[sandbox] Redis reachable")
    except Exception as exc:
        logger.error("[sandbox] Redis not reachable at %s:%s — %s: %s", _host, _port, type(exc).__name__, exc)
        raise HTTPException(status_code=503, detail="Worker unavailable — Redis is not reachable. Start Redis and try again.")

    logger.info("[sandbox] dispatching build_sandbox_task to Celery for %s…", request.env_name)
    try:
        from backend.app.worker.tasks import build_sandbox_task
        result = build_sandbox_task.delay(
            job_id=job_id,
            env_name=request.env_name,
            description=request.description,
            domain=request.domain,
            policy_requirements=request.policy_requirements,
            reward_requirements=request.reward_requirements,
        )
        logger.info("[sandbox] task queued — celery task_id=%s env_name=%s", result.id, request.env_name)
    except Exception:
        logger.exception("[sandbox] FAILED to dispatch task for %s", request.env_name)
        raise HTTPException(status_code=503, detail="Worker unavailable — could not queue build task")

    return {"job_id": job_id, "env_name": request.env_name}


@router.get("/", response_model=list[SandboxResponse])
def list_sandboxes(db: Session = Depends(get_db)):
    return (
        db.query(SandboxEnvironment)
        .filter(SandboxEnvironment.status.notin_(["deleted", "expired"]))
        .order_by(SandboxEnvironment.created_at.desc())
        .all()
    )


@router.get("/{env_name}", response_model=SandboxResponse)
def get_sandbox(env_name: str, db: Session = Depends(get_db)):
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    if sandbox.status == "running" and sandbox.container_id:
        try:
            import docker, docker.errors
            client = docker.from_env()
            container = client.containers.get(sandbox.container_id)
            container.reload()
            if container.status != "running":
                sandbox.status = "stopped"
                db.commit()
        except Exception:
            sandbox.status = "stopped"
            db.commit()
    return sandbox


@router.post("/{env_name}/stop", status_code=204)
def stop_sandbox(env_name: str, db: Session = Depends(get_db)):
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    if sandbox.container_id:
        try:
            from forge.envgen.container import ContainerRuntime
            ContainerRuntime().stop(sandbox.container_id)
        except Exception:
            pass
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
    db.delete(sandbox)
    db.commit()


@router.websocket("/ws/progress/{env_name}")
async def sandbox_progress(websocket: WebSocket, env_name: str):
    """Stream build progress from a Celery worker via Redis pub/sub."""
    logger.info("[ws:progress] client connected — env_name=%s", env_name)
    await websocket.accept()
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    channel = f"forge:progress:{env_name}"
    try:
        r = redis.asyncio.from_url(redis_url)
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        logger.info("[ws:progress] subscribed to Redis channel %s", channel)
    except Exception:
        logger.exception("[ws:progress] FAILED to connect to Redis (%s)", redis_url)
        await websocket.close(code=1011)
        return
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])
            logger.debug("[ws:progress] → %s: %s", env_name, data)
            await websocket.send_json(data)
            if data.get("done"):
                logger.info("[ws:progress] build done signal received for %s", env_name)
                break
    except WebSocketDisconnect:
        logger.info("[ws:progress] client disconnected — env_name=%s", env_name)
    except Exception:
        logger.exception("[ws:progress] unexpected error for %s", env_name)
    finally:
        await pubsub.unsubscribe(channel)
        await r.aclose()
        try:
            await websocket.close()
        except RuntimeError:
            pass  # client already closed the connection
        logger.info("[ws:progress] connection closed — env_name=%s", env_name)


@router.websocket("/ws/feed/{env_name}")
async def sandbox_event_feed(websocket: WebSocket, env_name: str, db: Session = Depends(get_db)):
    """Tail forge:events:<env_name> Redis Stream and push to frontend."""
    await websocket.accept()
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    from forge.envgen.telemetry.stream import StreamConsumer
    consumer = StreamConsumer(redis_url=redis_url, env_name=env_name)
    try:
        async for event in consumer.tail(last_id="$"):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        await consumer.close()
        await websocket.close()


@router.websocket("/ws/exec/{env_name}")
async def sandbox_exec(websocket: WebSocket, env_name: str, db: Session = Depends(get_db)):
    """Bridge WebSocket to docker exec shell for the sandbox container."""
    await websocket.accept()
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox or not sandbox.container_id:
        await websocket.send_text("Container not running\r\n")
        await websocket.close()
        return
    container_name = f"forge-{env_name}"
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "-i", container_name, "/bin/sh",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    async def read_output():
        while True:
            chunk = await proc.stdout.read(1024)
            if not chunk:
                break
            await websocket.send_text(chunk.decode(errors="replace"))

    async def write_input():
        try:
            while True:
                data = await websocket.receive_text()
                proc.stdin.write(data.encode())
                await proc.stdin.drain()
        except WebSocketDisconnect:
            pass

    try:
        await asyncio.gather(read_output(), write_input())
    finally:
        proc.kill()
        await websocket.close()
