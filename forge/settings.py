from __future__ import annotations

import os
from pathlib import Path

from forge.config import get_config


DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_DATABASE_URL = "sqlite:///./forge.db"
DEFAULT_GENERATED_ENVS_DIR = "generated_envs"
DEFAULT_SANDBOX_LIMIT = 10


def determinism_enabled() -> bool:
    """Return the experiment determinism mode (on by default)."""
    val = os.environ.get("FORGE_DETERMINISM")
    if val is not None:
        val = val.strip().lower()
        if val not in {"on", "off"}:
            raise ValueError("FORGE_DETERMINISM must be 'on' or 'off'")
        return val == "on"
    return get_config().evaluation.determinism_enabled


def experiment_seed(seed: int) -> int | None:
    """Drop configured seeds when running the determinism-off ablation."""
    return seed if determinism_enabled() else None


def redis_url() -> str:
    return os.environ.get("REDIS_URL", get_config().storage.redis_url)


def generated_envs_root() -> Path:
    return Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", get_config().storage.generated_envs_dir))


def database_url() -> str:
    return os.environ.get("FORGE_DB_URL", get_config().storage.database_url)


def sandbox_limit() -> int:
    return int(os.environ.get("FORGE_SANDBOX_LIMIT", str(get_config().sandbox.sandbox_limit)))
