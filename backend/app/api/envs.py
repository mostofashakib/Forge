# backend/app/api/envs.py
from __future__ import annotations
import shutil
from datetime import datetime, timezone
from pathlib import Path
from forge.settings import generated_envs_root
from forge.paths import confined_path
from forge.envgen.source_bundle import SourceBundleError, build_source_bundle
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models import CompileJob, SandboxEnvironment
from backend.app.services import episode_service
from forge.extraction.schemas import CompilerInput

router = APIRouter(prefix="/api/envs")


def _envs_root() -> Path:
    return generated_envs_root()


def _validate_env_name(env_name: str) -> None:
    if ".." in env_name or "/" in env_name or "\\" in env_name:
        raise HTTPException(status_code=400, detail="Invalid environment name")


class ConfigPayload(BaseModel):
    yaml: str


def _inactive_env_names(db: Session) -> set[str]:
    """Names of environments whose sandbox is deleted, expired, or past its TTL.

    These must not appear in active inventory even though their generated source
    directory still exists on disk (the expiry sweep tears down the container but
    leaves the files, and only runs periodically)."""
    rows = (
        db.query(SandboxEnvironment.id)
        .filter(
            or_(
                SandboxEnvironment.status.in_(["deleted", "expired"]),
                SandboxEnvironment.expires_at <= datetime.now(timezone.utc),
            )
        )
        .all()
    )
    return {row.id for row in rows}


@router.get("/", response_model=list[str])
def list_envs(db: Session = Depends(get_db)) -> list[str]:
    root = _envs_root()
    if not root.exists():
        return []
    inactive = _inactive_env_names(db)
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name not in inactive
    )


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


@router.get("/{env_name}/download")
def download_env_source(env_name: str) -> Response:
    """Download a generated environment as a standalone, runnable source zip."""
    _validate_env_name(env_name)
    try:
        env_dir = confined_path(_envs_root(), env_name)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid environment name")
    try:
        data = build_source_bundle(env_dir, env_name)
    except SourceBundleError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{env_name}.zip"'},
    )


@router.delete("/{env_name}", status_code=204)
def delete_env(env_name: str) -> None:
    _validate_env_name(env_name)
    env_dir = _envs_root() / env_name
    if not env_dir.exists():
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")
    shutil.rmtree(env_dir)


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
