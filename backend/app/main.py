from __future__ import annotations
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("FORGE_DEBUG") else logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.compile import router as compile_router
from backend.app.api.envs import router as envs_router
from backend.app.api.episodes import router as episodes_router
from backend.app.api.rollouts import router as rollouts_router
from backend.app.api.exports import router as exports_router
from backend.app.api.audit import router as audit_router
from backend.app.api.sandbox import router as sandbox_router
from backend.app.database import init_db

app = FastAPI(title="Forge API", version="0.3.0")

_cors_raw = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(compile_router)
app.include_router(envs_router)
app.include_router(episodes_router)
app.include_router(rollouts_router)
app.include_router(exports_router)
app.include_router(audit_router)
app.include_router(sandbox_router)


@app.on_event("startup")
def startup():
    logger.info("[startup] initialising database…")
    init_db()
    logger.info("[startup] database ready")
    _reattach_containers()
    logger.info("[startup] Forge API ready — CORS origins: %s", _cors_origins)


def _reattach_containers() -> None:
    try:
        from forge.envgen.container import ContainerRuntime
        from backend.app.database import get_session_factory
        from backend.app.models import SandboxEnvironment
        runtime = ContainerRuntime()
        managed = runtime.reattach_all()
        if not managed:
            logger.info("[startup] no running containers to reattach")
            return
        SessionLocal = get_session_factory()
        with SessionLocal() as db:
            for env_name, container_id, port in managed:
                sandbox = db.get(SandboxEnvironment, env_name)
                if sandbox:
                    sandbox.container_id = container_id
                    sandbox.container_port = port
                    sandbox.status = "running"
                    logger.info("[startup] reattached container for %s → port %s", env_name, port)
            db.commit()
    except Exception as exc:
        logger.warning("[startup] container reattach skipped — %s: %s", type(exc).__name__, exc)
