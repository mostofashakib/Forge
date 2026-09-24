from __future__ import annotations
import asyncio
import concurrent.futures
import json
import logging
import os
from datetime import datetime, timedelta, timezone
import uuid
import redis
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
import re
from typing import Literal
from urllib.parse import urlparse
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session
from backend.app.api._pubsub_relay import relay_pubsub
from backend.app.database import get_db, get_session_factory
from backend.app.docker_utils import is_docker_daemon_unavailable
from backend.app.models import SandboxEnvironment
from forge.settings import generated_envs_root, redis_url, sandbox_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sandbox")

DISPATCH_TIMEOUT_S = 15.0
# Bounds how long a request waits on Celery. A dispatch that outlives the
# timeout keeps its thread here, never one of the request threads.
_dispatch_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="sandbox-dispatch"
)
_BUILD_IN_PROGRESS = ("queued", "building")


class CreateSandboxRequest(BaseModel):
    env_name: str
    env_type: Literal["general", "cli", "browser", "premade:gmail", "premade:slack"] = "general"
    description: str = Field(default="", max_length=50_000)
    domain: str = "localhost"
    policy_requirements: str = Field(default="", max_length=20_000)
    reward_requirements: str = Field(default="", max_length=20_000)
    reference_urls: list[str] = Field(default_factory=list, max_length=5)
    use_user_researcher: bool = False
    # Generated environments are API-only unless a UI is explicitly requested.
    # Ignored for cli (already headless) and browser (inherently a UI).
    with_ui: bool = False
    source_product_name: str = Field(default="", max_length=200)
    source_product_url: str = Field(default="", max_length=2_000)
    ttl_days: int = Field(default=30, ge=1, le=365)
    # The simulated people who will share the environment with the agent. The
    # builder chooses who is in the world; nobody can be given an action here,
    # because the environment's actions do not exist until it is generated.
    personas: dict | None = None

    @field_validator("personas")
    @classmethod
    def validate_personas(cls, value: dict | None) -> dict | None:
        """Reject a malformed cast at the request, not mid-build.

        A build that fails twenty minutes in because a trait was out of range
        is a much worse error than a 422 here.
        """
        if value is None:
            return None
        from forge.personas.config import PersonaConfigError, dump_population, load_population

        try:
            population = load_population(value)
        except PersonaConfigError as exc:
            raise ValueError(str(exc)) from exc
        for spec in [*population.roster, *population.archetypes]:
            if spec.behavior.allowed_actions:
                raise ValueError(
                    f"persona '{spec.profile.id}' cannot be granted actions at "
                    "creation time — the environment's actions do not exist "
                    "yet. Choose what each person can do on the Simulated "
                    "People page once the build finishes."
                )
        return dump_population(population)

    @field_validator("env_name")
    @classmethod
    def validate_env_name(cls, v: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", v):
            raise ValueError(
                "env_name must start with a letter or digit and contain only "
                "letters, digits, underscores, and hyphens (no spaces)"
            )
        return v

    @model_validator(mode="after")
    def normalize_ui_selection(self) -> CreateSandboxRequest:
        """The UI toggle governs generated apps only.

        Premade replicas always ship a UI and browser sandboxes are a browser;
        a CLI sandbox is a shell with no page at all. Only "general" builds
        actually run (or skip) the UI specialist.
        """
        if self.env_type != "general":
            self.with_ui = self.env_type != "cli"
        return self

    @model_validator(mode="after")
    def validate_research_source(self) -> CreateSandboxRequest:
        self.source_product_name = self.source_product_name.strip()
        self.source_product_url = self.source_product_url.strip()
        if not self.use_user_researcher:
            self.source_product_name = ""
            self.source_product_url = ""
            return self
        if not re.fullmatch(r"https?://[^\s]+", self.source_product_url):
            raise ValueError(
                "source_product_url must be a valid http or https URL when user research is enabled"
            )
        if not self.source_product_name:
            parsed = urlparse(self.source_product_url)
            hostname = (parsed.hostname or "").removeprefix("www.")
            source_key = hostname.split(".")[0]
            if hostname in {"github.com", "gitlab.com", "bitbucket.org"}:
                path_parts = [part for part in parsed.path.split("/") if part]
                if path_parts:
                    source_key = path_parts[-1].removesuffix(".git")
            self.source_product_name = re.sub(r"[-_]+", " ", source_key).strip().title()
        return self


class SandboxResponse(BaseModel):
    id: str
    status: str
    env_type: str = "general"
    has_ui: bool = True
    container_id: str | None = None
    container_port: int | None = None
    ttl_days: int
    expires_at: datetime
    created_at: datetime
    policy_requirements: str | None = None
    reward_requirements: str | None = None

    model_config = {"from_attributes": True}


class SandboxCapacityResponse(BaseModel):
    active_count: int
    limit: int


def _active_criteria() -> list:
    """Predicate for a genuinely active sandbox.

    A sandbox is inactive if it was deleted/expired *or* its TTL has already
    lapsed — even when the periodic cleanup sweep has not yet flipped its
    status, a time-expired environment must never appear in active inventory.
    """
    return [
        SandboxEnvironment.status.notin_(["deleted", "expired"]),
        SandboxEnvironment.expires_at > datetime.now(timezone.utc),
    ]


def _active_sandbox_count(db: Session) -> int:
    return db.query(SandboxEnvironment).filter(*_active_criteria()).count()


def _remove_undispatched_sandbox(db: Session, sandbox: SandboxEnvironment) -> None:
    """Remove a build record when Celery definitively rejected the dispatch."""
    db.delete(sandbox)
    db.commit()


@router.post("/", status_code=202)
def create_sandbox(request: CreateSandboxRequest, db: Session = Depends(get_db)):
    logger.info("[sandbox] POST /api/sandbox/ — env_name=%s", request.env_name)

    active_count = _active_sandbox_count(db)
    max_environments = sandbox_limit()
    if active_count >= max_environments:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Environment limit reached ({max_environments} max). "
                "Delete an existing environment to create a new one."
            ),
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

    redis_connection_url = redis_url()
    logger.info("[sandbox] checking Redis at %s…", redis_connection_url)
    try:
        pong = redis.from_url(
            redis_connection_url, socket_connect_timeout=2, socket_timeout=2
        ).ping()
        if not pong:
            raise RuntimeError("PING returned false")
        logger.info("[sandbox] Redis healthy")
    except Exception as exc:
        logger.error("[sandbox] Redis health check failed — %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=503,
            detail="Worker unavailable — Redis is not responding. Run: redis-server --daemonize yes",
        )

    # The worker needs this row before it can start, so persist it only after
    # Redis is known to be available and immediately before dispatch.
    job_id = str(uuid.uuid4())
    sandbox = SandboxEnvironment(
        id=request.env_name,
        status="queued",
        env_type=request.env_type,
        ttl_days=request.ttl_days,
        expires_at=datetime.now(timezone.utc) + timedelta(days=request.ttl_days),
        policy_requirements=request.policy_requirements or None,
        reward_requirements=request.reward_requirements or None,
        has_ui=request.with_ui,
    )
    db.add(sandbox)
    db.commit()
    logger.info("[sandbox] DB row created for %s (job_id=%s)", request.env_name, job_id)

    logger.info("[sandbox] dispatching build_sandbox_task to Celery for %s…", request.env_name)
    from backend.app.worker.tasks import build_sandbox_task
    dispatch = _dispatch_pool.submit(
        lambda: build_sandbox_task.delay(
            job_id=job_id,
            env_name=request.env_name,
            env_type=request.env_type,
            description=request.description,
            domain=request.domain,
            policy_requirements=request.policy_requirements,
            reward_requirements=request.reward_requirements,
            reference_urls=request.reference_urls,
            use_user_researcher=request.use_user_researcher,
            with_ui=request.with_ui,
            source_product_name=request.source_product_name,
            source_product_url=request.source_product_url,
            personas=request.personas,
        )
    )
    try:
        result = dispatch.result(timeout=DISPATCH_TIMEOUT_S)
        logger.info("[sandbox] task queued — celery task_id=%s env_name=%s", result.id, request.env_name)
    except concurrent.futures.TimeoutError:
        logger.error("[sandbox] Celery dispatch timed out for %s", request.env_name)
        # The dispatch thread may still complete after our timeout. Preserve
        # the row for a late worker, but never leave it falsely queued.
        sandbox.status = "error"
        db.commit()
        raise HTTPException(status_code=503, detail=f"Worker unavailable — Celery did not accept the task within {DISPATCH_TIMEOUT_S:g} s. Check Redis and the Celery worker.")
    except Exception:
        logger.exception("[sandbox] FAILED to dispatch task for %s", request.env_name)
        _remove_undispatched_sandbox(db, sandbox)
        raise HTTPException(status_code=503, detail="Worker unavailable — could not queue build task")

    return {"job_id": job_id, "env_name": request.env_name}


@router.get("/", response_model=list[SandboxResponse])
def list_sandboxes(db: Session = Depends(get_db)):
    return (
        db.query(SandboxEnvironment)
        .filter(*_active_criteria())
        .order_by(SandboxEnvironment.created_at.desc())
        .all()
    )


@router.get("/capacity", response_model=SandboxCapacityResponse)
def get_sandbox_capacity(db: Session = Depends(get_db)):
    return SandboxCapacityResponse(
        active_count=_active_sandbox_count(db),
        limit=sandbox_limit(),
    )


@router.get("/{env_name}", response_model=SandboxResponse)
def get_sandbox(env_name: str, db: Session = Depends(get_db)):
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    if sandbox.status == "running" and sandbox.container_id:
        import docker, docker.errors
        client = None
        try:
            client = docker.from_env()
            _sync_with_container(sandbox, client.containers.get(sandbox.container_id), db)
        except docker.errors.NotFound:
            sandbox.status = "stopped"
            db.commit()
        except Exception as exc:
            if is_docker_daemon_unavailable(exc):
                logger.debug(
                    "[sandbox:get] Docker is not running; returning persisted status for %s",
                    env_name,
                )
            else:
                logger.warning(
                    "[sandbox:get] could not synchronize Docker status for %s; "
                    "returning persisted status: %s: %s",
                    env_name,
                    type(exc).__name__,
                    exc,
                )
        finally:
            if client is not None:
                client.close()
    return sandbox


def _sync_with_container(sandbox: SandboxEnvironment, container, db: Session) -> None:
    """Heal the persisted status and port from what Docker reports right now."""
    container.reload()
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
                    sandbox.id, port_key, container.ports,
                )
                sandbox.status = "stopped"
                db.commit()


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
    import docker, docker.errors
    client = None
    try:
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
    finally:
        if client is not None:
            client.close()


@router.post("/{env_name}/stop", status_code=204)
def stop_sandbox(env_name: str, db: Session = Depends(get_db)):
    sandbox = db.get(SandboxEnvironment, env_name)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    if sandbox.container_id:
        try:
            from forge.envgen.container import ContainerRuntime
            ContainerRuntime().stop(sandbox.container_id)
        except Exception as exc:
            logger.exception("[sandbox:stop] failed for %s", env_name)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to stop container: {exc}",
            ) from exc
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
    env_dir = generated_envs_root() / env_name
    if env_dir.exists():
        shutil.rmtree(env_dir)
    db.delete(sandbox)
    db.commit()


def _finished_build_message(env_name: str) -> dict | None:
    """The terminal progress message for a build that already ended, else None."""
    with get_session_factory()() as db:
        sandbox = db.get(SandboxEnvironment, env_name)
        if sandbox is None:
            return {"done": True, "error": f"Sandbox '{env_name}' not found"}
        if sandbox.status in _BUILD_IN_PROGRESS:
            return None
        if sandbox.status == "error":
            return {"done": True, "error": "Build failed — check the worker logs for details."}
        return {"done": True}


def _container_id(env_name: str) -> str | None:
    """Look up the container in a short session, not one held for a socket's lifetime."""
    with get_session_factory()() as db:
        sandbox = db.get(SandboxEnvironment, env_name)
        return sandbox.container_id if sandbox else None


@router.websocket("/ws/progress/{env_name}")
async def sandbox_progress(websocket: WebSocket, env_name: str):
    """Stream build progress from a Celery worker via Redis pub/sub."""
    logger.info("[ws:progress] client connected — env_name=%s", env_name)
    await websocket.accept()
    redis_connection_url = redis_url()
    channel = f"forge:progress:{env_name}"
    try:
        r = redis.asyncio.from_url(redis_connection_url)
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        logger.info("[ws:progress] subscribed to Redis channel %s", channel)
    except Exception:
        logger.exception("[ws:progress] FAILED to connect to Redis (%s)", redis_connection_url)
        await websocket.close(code=1011)
        return
    try:
        # Checked after subscribing, so a build that ends in between still
        # reaches this client through the channel.
        finished = await run_in_threadpool(_finished_build_message, env_name)
        if finished is not None:
            await websocket.send_json(finished)
        else:
            await relay_pubsub(websocket, pubsub, is_final=lambda data: bool(data.get("done")))
        logger.info("[ws:progress] build done for %s", env_name)
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
async def sandbox_event_feed(websocket: WebSocket, env_name: str):
    """Tail forge:events:<env_name> Redis Stream and push to frontend."""
    try:
        await websocket.accept()
    except (WebSocketDisconnect, RuntimeError):
        # The React client can mount and immediately unmount the feed before
        # Uvicorn completes the handshake. That is a normal disconnect race,
        # not an application failure.
        logger.debug("[ws:feed] client disconnected before accept — env_name=%s", env_name)
        return

    redis_connection_url = redis_url()
    from forge.envgen.telemetry.stream import StreamConsumer
    consumer = StreamConsumer(redis_url=redis_connection_url, env_name=env_name)
    try:
        async for event in consumer.tail(last_id="$"):
            try:
                await websocket.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                logger.debug("[ws:feed] client disconnected — env_name=%s", env_name)
                break
    except WebSocketDisconnect:
        logger.debug("[ws:feed] client disconnected — env_name=%s", env_name)
    finally:
        await consumer.close()
        try:
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass


@router.websocket("/ws/exec/{env_name}")
async def sandbox_exec(websocket: WebSocket, env_name: str):
    """Bridge WebSocket to docker exec shell via PTY for full interactive terminal support."""
    await websocket.accept()
    container_id = await run_in_threadpool(_container_id, env_name)
    if not container_id:
        await websocket.send_text("Container not running\r\n")
        await websocket.close()
        return

    import fcntl
    import pty as _pty
    import struct
    import subprocess as _subprocess
    import termios

    master_fd, slave_fd = _pty.openpty()
    # Default terminal size: 80 cols × 24 rows
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))

    proc = _subprocess.Popen(
        ["docker", "exec", "-it", container_id, "/bin/bash"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    loop = asyncio.get_running_loop()
    closed = asyncio.Event()

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
                        if not (1 <= cols <= 500 and 1 <= rows <= 200):
                            continue
                        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
                        continue
                except (ValueError, KeyError, TypeError):
                    pass
                try:
                    os.write(master_fd, text.encode())
                except OSError:
                    break
        except Exception:
            pass
        finally:
            closed.set()

    ws_task = asyncio.create_task(_ws_reader())
    try:
        await closed.wait()
    finally:
        ws_task.cancel()
        loop.remove_reader(master_fd)
        try:
            os.close(master_fd)
        except OSError:
            pass
        proc.terminate()
        # The reaping thread finishes even if this handler is cancelled mid-await.
        await asyncio.to_thread(proc.wait)
    try:
        await websocket.close()
    except RuntimeError:
        pass


@router.websocket("/ws/activity/{env_name}")
async def sandbox_activity(websocket: WebSocket, env_name: str):
    """Stream container logs to the Observability panel."""
    await websocket.accept()
    container_id = await run_in_threadpool(_container_id, env_name)
    closed = asyncio.Event()

    async def _docker_logs() -> None:
        if not container_id:
            return
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "logs", "--follow", "--timestamps", container_id,
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
                except (WebSocketDisconnect, RuntimeError):
                    break
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                "[sandbox:activity] Docker log stream failed for %s: %s",
                env_name,
                exc,
            )
        finally:
            if proc is not None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    logger.debug("[sandbox:activity] Docker log process already exited for %s", env_name)
                await proc.wait()

    async def _ws_watcher() -> None:
        try:
            while True:
                await websocket.receive_text()
        except (WebSocketDisconnect, Exception):
            closed.set()

    tasks = [
        asyncio.create_task(_docker_logs()),
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
