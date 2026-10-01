from __future__ import annotations
import concurrent.futures
import logging
from datetime import datetime, timedelta, timezone
import uuid
import redis
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.api.sandbox_schemas import (
    CreateSandboxRequest,
    SandboxCapacityResponse,
    SandboxResponse,
)
from backend.app.database import get_db
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
    from backend.app.worker.sandbox_build_tasks import build_sandbox_task
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
            _sync_with_container(sandbox, client.containers.get(sandbox.container_id), db, client)
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


def _sync_with_container(sandbox: SandboxEnvironment, container, db: Session, client) -> None:
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
            from forge.envgen.container import ContainerRuntime
            # App containers publish nothing. Their port is their gateway's.
            live_port = ContainerRuntime(client).host_port(container) or 0
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
                    "[sandbox:get] %s container running but no host port "
                    "(container.ports=%s) — demoting to stopped so user can restart",
                    sandbox.id, container.ports,
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


