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
import re
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models import SandboxEnvironment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sandbox")


class CreateSandboxRequest(BaseModel):
    env_name: str
    env_type: str = "general"   # "general" | "cli" | "browser"
    description: str = ""
    domain: str = "localhost"
    policy_requirements: str = ""
    reward_requirements: str = ""
    ttl_days: int = 30

    @field_validator("env_name")
    @classmethod
    def validate_env_name(cls, v: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", v):
            raise ValueError(
                "env_name must start with a letter or digit and contain only "
                "letters, digits, underscores, and hyphens (no spaces)"
            )
        return v


class SandboxResponse(BaseModel):
    id: str
    status: str
    env_type: str = "general"
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

    active_count = (
        db.query(SandboxEnvironment)
        .filter(SandboxEnvironment.status.notin_(["deleted", "expired"]))
        .count()
    )
    if active_count >= 10:
        raise HTTPException(
            status_code=429,
            detail="Environment limit reached (10 max). Delete an existing environment to create a new one.",
        )

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
        env_type=request.env_type,
        ttl_days=request.ttl_days,
        expires_at=datetime.now(timezone.utc) + timedelta(days=request.ttl_days),
        policy_requirements=request.policy_requirements or None,
        reward_requirements=request.reward_requirements or None,
    )
    db.add(sandbox)
    db.commit()
    logger.info("[sandbox] DB row created for %s (job_id=%s)", request.env_name, job_id)

    import asyncio
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    logger.info("[sandbox] checking Redis at %s…", redis_url)
    try:
        loop = asyncio.get_running_loop()
        pong = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: redis.from_url(
                    redis_url, socket_connect_timeout=2, socket_timeout=2
                ).ping(),
            ),
            timeout=4.0,
        )
        if not pong:
            raise RuntimeError("PING returned false")
        logger.info("[sandbox] Redis healthy")
    except Exception as exc:
        logger.error("[sandbox] Redis health check failed — %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=503,
            detail="Worker unavailable — Redis is not responding. Run: redis-server --daemonize yes",
        )

    logger.info("[sandbox] dispatching build_sandbox_task to Celery for %s…", request.env_name)
    try:
        from backend.app.worker.tasks import build_sandbox_task
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: build_sandbox_task.delay(
                    job_id=job_id,
                    env_name=request.env_name,
                    env_type=request.env_type,
                    description=request.description,
                    domain=request.domain,
                    policy_requirements=request.policy_requirements,
                    reward_requirements=request.reward_requirements,
                ),
            ),
            timeout=15.0,
        )
        logger.info("[sandbox] task queued — celery task_id=%s env_name=%s", result.id, request.env_name)
    except asyncio.TimeoutError:
        logger.error("[sandbox] Celery dispatch timed out for %s", request.env_name)
        raise HTTPException(status_code=503, detail="Worker unavailable — Celery did not accept the task within 15 s. Check Redis and the Celery worker.")
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
            state = container.attrs.get("State", {}) or {}
            # A container that's been respawned by Docker's restart policy is
            # crashing on boot (LLM-generated app likely has a bug). It looks
            # "running" only momentarily between crashes — flag it as error
            # so the UI stops claiming it's healthy.
            restart_count = container.attrs.get("RestartCount", 0) or 0
            if container.status == "restarting" or restart_count > 0:
                sandbox.status = "error"
                db.commit()
            elif container.status != "running":
                sandbox.status = "stopped"
                db.commit()
            else:
                # Resync container_port from the live container — heals the
                # DB-says-running-but-port-is-null state that can happen after
                # a host reboot, a half-failed /start, or worker reattach.
                # CLI envs intentionally have no HTTP port, so leave them alone.
                if sandbox.image_tag != "builtin:cli":
                    port_key = "3000/tcp" if sandbox.image_tag == "builtin:browser" else "8000/tcp"
                    bindings = container.ports.get(port_key) or []
                    live_port = 0
                    if bindings and isinstance(bindings, list):
                        host_port = bindings[0].get("HostPort") if isinstance(bindings[0], dict) else None
                        try:
                            live_port = int(host_port) if host_port else 0
                        except (TypeError, ValueError):
                            live_port = 0
                    if live_port > 0:
                        if sandbox.container_port != live_port:
                            sandbox.container_port = live_port
                            db.commit()
                    elif not sandbox.container_port:
                        # Container is up but the port mapping doesn't exist
                        # (or didn't survive a daemon restart). Demote to
                        # "stopped" so the UI shows a Start button — /start
                        # will run a fresh container with a real port binding.
                        logger.warning(
                            "[sandbox:get] %s container running but no %s binding "
                            "(container.ports=%s) — demoting to stopped so user can restart",
                            env_name, port_key, container.ports,
                        )
                        sandbox.status = "stopped"
                        db.commit()
        except docker.errors.NotFound:
            sandbox.status = "stopped"
            db.commit()
        except Exception:
            pass  # SDK/credential error — trust the DB status
    return sandbox


@router.post("/{env_name}/start")
def start_sandbox(env_name: str, db: Session = Depends(get_db)):
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    # Only short-circuit when the env is genuinely healthy: status=running AND
    # we have a host port (or it's a CLI env which has none by design).
    # Otherwise fall through to runtime.start(), which will refresh the
    # container and resync the port. This is what fixes the
    # "running but iframe says not running" state from the user's side —
    # clicking Start now actually does something.
    is_cli = sandbox.image_tag == "builtin:cli"
    if sandbox.status == "running" and (is_cli or sandbox.container_port):
        return {"status": "running", "container_port": sandbox.container_port}
    if not sandbox.image_tag:
        raise HTTPException(status_code=409, detail="No image available — environment must be rebuilt")
    try:
        from forge.envgen.container import ContainerRuntime
        runtime = ContainerRuntime()
        container_id, port = runtime.start(
            env_name=env_name,
            container_id=sandbox.container_id or "",
            image_tag=sandbox.image_tag,
        )
        sandbox.container_id = container_id
        sandbox.container_port = port
        sandbox.status = "running"
        db.commit()
        return {"status": "running", "container_port": port}
    except RuntimeError as exc:
        # Image-missing path from runtime.start — DB has stale image_tag from a
        # previous build. Clear it so the next /start returns 409 cleanly and
        # the UI prompts a rebuild.
        logger.warning("[sandbox:start] %s — clearing stale image_tag (%s)", env_name, exc)
        sandbox.image_tag = None
        sandbox.container_id = None
        sandbox.container_port = None
        sandbox.status = "stopped"
        db.commit()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[sandbox:start] failed for %s", env_name)
        raise HTTPException(status_code=500, detail=f"Failed to start container: {exc}") from exc


@router.get("/{env_name}/logs")
def get_sandbox_logs(env_name: str, tail: int = 200, db: Session = Depends(get_db)):
    """Return the last `tail` lines of the container's combined stdout+stderr.

    Crucial for debugging crash-loops where the container won't stay up — the
    logs surface the actual error (ModuleNotFoundError, port-in-use, etc.)
    that the LLM-generated app hit on boot.
    """
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    if not sandbox.container_id:
        return {"logs": "", "exit_code": None, "restart_count": 0}
    try:
        import docker, docker.errors
        client = docker.from_env()
        container = client.containers.get(sandbox.container_id)
        container.reload()
        raw = container.logs(tail=tail, stdout=True, stderr=True, timestamps=False)
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        state = container.attrs.get("State", {}) or {}
        return {
            "logs": text,
            "status": container.status,
            "exit_code": state.get("ExitCode"),
            "restart_count": container.attrs.get("RestartCount", 0) or 0,
            "error": state.get("Error") or None,
        }
    except docker.errors.NotFound:
        raise HTTPException(status_code=410, detail="Container no longer exists — environment must be rebuilt") from None
    except Exception as exc:
        logger.exception("[sandbox:logs] failed for %s", env_name)
        raise HTTPException(status_code=500, detail=f"Failed to read container logs: {exc}") from exc


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
    """Bridge WebSocket to docker exec shell via PTY for full interactive terminal support."""
    await websocket.accept()
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox or not sandbox.container_id:
        await websocket.send_text("Container not running\r\n")
        await websocket.close()
        return

    container_name = f"forge-{env_name}"

    import fcntl
    import pty as _pty
    import struct
    import subprocess as _subprocess
    import termios

    master_fd, slave_fd = _pty.openpty()
    # Default terminal size: 80 cols × 24 rows
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))

    proc = _subprocess.Popen(
        ["docker", "exec", "-it", container_name, "/bin/bash"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    loop = asyncio.get_running_loop()
    closed = asyncio.Event()

    cmd_buf: list[str] = []

    async def _publish_command(cmd: str) -> None:
        if not cmd.strip():
            return
        try:
            r_act = redis.asyncio.from_url(redis_url)
            await r_act.xadd(
                f"forge:activity:{env_name}",
                {"ts": datetime.now(timezone.utc).isoformat(), "type": "command", "content": cmd.strip()},
                maxlen=500,
            )
            await r_act.aclose()
        except Exception:
            pass

    def _read_pty() -> None:
        try:
            data = os.read(master_fd, 4096)
            loop.create_task(_forward(data))
        except OSError:
            closed.set()

    async def _forward(data: bytes) -> None:
        try:
            await websocket.send_text(data.decode(errors="replace"))
        except Exception:
            closed.set()

    loop.add_reader(master_fd, _read_pty)

    async def _ws_reader() -> None:
        try:
            while True:
                text = await websocket.receive_text()
                # Resize message: {"type":"resize","cols":N,"rows":N}
                try:
                    msg = json.loads(text)
                    if msg.get("type") == "resize":
                        cols, rows = int(msg["cols"]), int(msg["rows"])
                        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
                        continue
                except (ValueError, KeyError, TypeError):
                    pass
                # Buffer keystrokes for command observability
                for ch in text:
                    if ch in ('\r', '\n'):
                        await _publish_command(''.join(cmd_buf))
                        cmd_buf.clear()
                    elif ch in ('\x7f', '\x08'):  # backspace
                        if cmd_buf:
                            cmd_buf.pop()
                    elif ch == '\x03':  # Ctrl+C
                        cmd_buf.clear()
                    elif len(ch) == 1 and ord(ch) >= 32:
                        cmd_buf.append(ch)
                try:
                    os.write(master_fd, text.encode())
                except OSError:
                    break
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            closed.set()

    ws_task = asyncio.create_task(_ws_reader())
    await closed.wait()
    ws_task.cancel()

    loop.remove_reader(master_fd)
    try:
        os.close(master_fd)
    except OSError:
        pass
    proc.terminate()
    try:
        await websocket.close()
    except RuntimeError:
        pass


@router.websocket("/ws/activity/{env_name}")
async def sandbox_activity(websocket: WebSocket, env_name: str, db: Session = Depends(get_db)):
    """Stream container logs + CLI command events to the Observability panel."""
    await websocket.accept()
    sandbox = db.get(SandboxEnvironment, env_name)
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    closed = asyncio.Event()

    async def _docker_logs() -> None:
        if not sandbox or not sandbox.container_id:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "logs", "--follow", "--timestamps", sandbox.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            while not closed.is_set():
                try:
                    raw = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip()
                # docker --timestamps format: "2024-01-15T10:30:45.123456789Z content"
                ts, _, content = line.partition(" ")
                if not content:
                    content, ts = ts, ""
                try:
                    await websocket.send_json({"type": "log", "ts": ts[:19], "content": content})
                except Exception:
                    break
        except Exception:
            pass
        finally:
            try:
                proc.terminate()
            except Exception:
                pass

    async def _redis_events() -> None:
        try:
            r = redis.asyncio.from_url(redis_url)
            last_id = "$"
            while not closed.is_set():
                results = await r.xread({f"forge:activity:{env_name}": last_id}, block=500, count=20)
                for _, messages in (results or []):
                    for msg_id, fields in messages:
                        last_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                        evt = {
                            (k.decode() if isinstance(k, bytes) else k):
                            (v.decode() if isinstance(v, bytes) else v)
                            for k, v in fields.items()
                        }
                        if not closed.is_set():
                            try:
                                await websocket.send_json(evt)
                            except Exception:
                                closed.set()
            await r.aclose()
        except Exception:
            pass

    async def _ws_watcher() -> None:
        try:
            while True:
                await websocket.receive_text()
        except (WebSocketDisconnect, Exception):
            closed.set()

    tasks = [
        asyncio.create_task(_docker_logs()),
        asyncio.create_task(_redis_events()),
        asyncio.create_task(_ws_watcher()),
    ]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    closed.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    try:
        await websocket.close()
    except Exception:
        pass
