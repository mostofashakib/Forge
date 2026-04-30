from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.compile import router as compile_router
from backend.app.api.envs import router as envs_router
from backend.app.api.episodes import router as episodes_router
from backend.app.api.rollouts import router as rollouts_router
from backend.app.api.exports import router as exports_router
from backend.app.api.audit import router as audit_router
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


@app.on_event("startup")
def startup():
    init_db()
