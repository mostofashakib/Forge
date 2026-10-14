"""GET /api/health: whether every dependency Forge relies on works.

Docker's health check, the uptime heartbeat, smoke.sh, and the deploy pipeline
all read this one answer. Each check returns "ok" (or "disabled") or raises,
and a raised error is reported as its text with any connection secret removed.
"""
from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.app.services.ollama_catalog import list_ollama_models
from backend.app.services.settings_registry import DEFAULT_OLLAMA_URL
from forge.credentials import PROVIDER_CREDENTIALS, is_configured
from forge.extraction.llm_client import generation_models
from forge.settings import database_url, redis_url

router = APIRouter(prefix="/api/health", tags=["health"])

CHECK_TIMEOUT_S = 2.0
# Writes Litestream has not copied for longer than this are data at risk.
MAX_BACKUP_LAG_S = 600
PASSING = ("ok", "disabled")

_URL_USERINFO = re.compile(r"(\b[a-z][a-z0-9+.-]*://)[^/@\s]*@", re.IGNORECASE)


def describe_error(exc: BaseException) -> str:
    """The error's text, with credentials inside any URL replaced."""
    message = str(exc) or type(exc).__name__
    return _URL_USERINFO.sub(r"\1***@", message)[:500]


def check_database() -> str:
    from backend.app.database import get_engine

    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    return "ok"


def check_redis() -> str:
    import redis

    client = redis.from_url(redis_url(), socket_connect_timeout=CHECK_TIMEOUT_S, socket_timeout=CHECK_TIMEOUT_S)
    try:
        client.ping()
    finally:
        client.close()
    return "ok"


def check_docker() -> str:
    import docker

    client = docker.from_env(timeout=int(CHECK_TIMEOUT_S))
    try:
        client.ping()
    finally:
        client.close()
    return "ok"


def check_worker() -> str:
    from backend.app.worker.celery_app import celery

    replies = celery.control.inspect(timeout=CHECK_TIMEOUT_S / 2).ping()
    if not replies:
        raise RuntimeError("no Celery worker answered a ping")
    return "ok"


def check_llm() -> str:
    """The configured generation provider is usable, without a paid call."""
    provider = os.environ.get("FORGE_LLM_PROVIDER", "anthropic").lower()
    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL
        pulled = {model["name"] for model in list_ollama_models(base_url)}
        missing = [name for name in generation_models() if name not in pulled]
        if missing:
            raise RuntimeError(f"Ollama at {base_url} has not pulled: {', '.join(missing)}")
        return "ok"
    credential = PROVIDER_CREDENTIALS.get(provider)
    if credential is None:
        raise RuntimeError(f"unknown FORGE_LLM_PROVIDER {provider!r}")
    if not is_configured(credential):
        raise RuntimeError(f"{credential.env_vars[0]} is not configured")
    return "ok"


def _sqlite_path() -> Path:
    url = database_url()
    if not url.startswith("sqlite:///"):
        raise RuntimeError("FORGE_LITESTREAM_DIR is set but FORGE_DB_URL is not a SQLite file")
    return Path(url.removeprefix("sqlite:///"))


def check_backup() -> str:
    """Litestream lag: how long the database has held writes Litestream has not copied.

    An idle database with old Litestream files is fine, since nothing is at risk.
    """
    litestream_dir = os.environ.get("FORGE_LITESTREAM_DIR")
    if not litestream_dir:
        return "disabled"
    stamps = [path.stat().st_mtime for path in Path(litestream_dir).rglob("*") if path.is_file()]
    if not stamps:
        raise RuntimeError(f"no Litestream files under {litestream_dir}")
    db = _sqlite_path()
    wal = db.with_name(db.name + "-wal")
    last_write = (wal if wal.exists() else db).stat().st_mtime
    lag = last_write - max(stamps)
    if lag > MAX_BACKUP_LAG_S:
        raise RuntimeError(f"database writes not copied by Litestream for {int(lag)} s")
    return "ok"


CHECKS: dict[str, Callable[[], str]] = {
    "database": check_database,
    "redis": check_redis,
    "docker": check_docker,
    "worker": check_worker,
    "llm": check_llm,
    "backup": check_backup,
}


async def _run(check: Callable[[], str]) -> str:
    try:
        return await asyncio.wait_for(asyncio.to_thread(check), timeout=CHECK_TIMEOUT_S)
    except TimeoutError:
        return f"timed out after {CHECK_TIMEOUT_S:g} s"
    except Exception as exc:  # noqa: BLE001 — every failure is reported, none crash the endpoint
        return describe_error(exc)


@router.get("")
async def health() -> JSONResponse:
    names = list(CHECKS)
    results = await asyncio.gather(*(_run(CHECKS[name]) for name in names))
    checks = dict(zip(names, results))
    healthy = all(result in PASSING for result in checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "checks": checks},
    )
