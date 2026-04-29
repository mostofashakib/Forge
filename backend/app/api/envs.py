from __future__ import annotations
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/envs")


def _envs_root() -> Path:
    return Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))


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
    config_path = _envs_root() / env_name / "custom" / "config.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found for '{env_name}'")
    return ConfigPayload(yaml=config_path.read_text())


@router.put("/{env_name}/config", response_model=ConfigPayload)
def update_config(env_name: str, payload: ConfigPayload) -> ConfigPayload:
    custom_dir = _envs_root() / env_name / "custom"
    if not custom_dir.exists():
        raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")
    config_path = custom_dir / "config.yaml"
    config_path.write_text(payload.yaml)
    return ConfigPayload(yaml=payload.yaml)
