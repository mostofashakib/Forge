# backend/app/api/envs.py
from __future__ import annotations
import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models import CompileJob
from backend.app.services import episode_service
from forge.extraction.schemas import CompilerInput

router = APIRouter(prefix="/api/envs")


def _envs_root() -> Path:
    return Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))


def _validate_env_name(env_name: str) -> None:
    if ".." in env_name or "/" in env_name or "\\" in env_name:
        raise HTTPException(status_code=400, detail="Invalid environment name")


class ConfigPayload(BaseModel):
    yaml: str


@router.get("/", response_model=list[str])
def list_envs() -> list[str]:
    root = _envs_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


@router.get("/{env_name}/config", response_model=ConfigPayload)
def get_config(env_name: str) -> ConfigPayload:
    _validate_env_name(env_name)
    config_path = _envs_root() / env_name / "custom" / "config.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found for '{env_name}'")
    return ConfigPayload(yaml=config_path.read_text())


@router.put("/{env_name}/config", response_model=ConfigPayload)
def update_config(env_name: str, payload: ConfigPayload) -> ConfigPayload:
    _validate_env_name(env_name)
    custom_dir = _envs_root() / env_name / "custom"
    if not custom_dir.exists():
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")
    config_path = custom_dir / "config.yaml"
    config_path.write_text(payload.yaml)
    return ConfigPayload(yaml=payload.yaml)


@router.get("/{env_name}/stats")
def get_env_stats(env_name: str, db: Session = Depends(get_db)):
    _validate_env_name(env_name)
    return episode_service.get_stats(env_name, db)


@router.get("/{env_name}/compiler-input")
def get_compiler_input(env_name: str, db: Session = Depends(get_db)):
    _validate_env_name(env_name)
    job = (
        db.query(CompileJob)
        .filter_by(project_name=env_name)
        .order_by(CompileJob.created_at.desc())
        .first()
    )
    if not job or not job.compiler_input_json:
        raise HTTPException(status_code=404, detail=f"No compiler input found for '{env_name}'")
    return CompilerInput.model_validate_json(job.compiler_input_json)
