from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.compile import router as compile_router
from backend.app.api.envs import router as envs_router
from backend.app.api.episodes import router as episodes_router
from backend.app.database import init_db

app = FastAPI(title="Forge API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(compile_router)
app.include_router(envs_router)
app.include_router(episodes_router)


@app.on_event("startup")
def startup():
    init_db()
