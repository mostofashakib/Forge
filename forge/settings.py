from __future__ import annotations

import os
from pathlib import Path


DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_DATABASE_URL = "sqlite:///./forge.db"
DEFAULT_GENERATED_ENVS_DIR = "generated_envs"
DEFAULT_SANDBOX_LIMIT = 10


def determinism_enabled() -> bool:
    """Return the experiment determinism mode (on by default)."""
    value = os.environ.get("FORGE_DETERMINISM", "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError("FORGE_DETERMINISM must be 'on' or 'off'")
    return value == "on"


def experiment_seed(seed: int) -> int | None:
    """Drop configured seeds when running the determinism-off ablation."""
    return seed if determinism_enabled() else None


def redis_url() -> str:
    return os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)


def generated_envs_root() -> Path:
    return Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", DEFAULT_GENERATED_ENVS_DIR))


def database_url() -> str:
    return os.environ.get("FORGE_DB_URL", DEFAULT_DATABASE_URL)


def sandbox_limit() -> int:
    return int(os.environ.get("FORGE_SANDBOX_LIMIT", str(DEFAULT_SANDBOX_LIMIT)))
