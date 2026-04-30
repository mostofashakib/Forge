from __future__ import annotations
import importlib
import os
import sys
from pathlib import Path


def load_forge_env(env_name: str, telemetry):
    """Dynamically import a generated ForgeEnv and inject telemetry."""
    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
    parent = str(envs_root.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"generated_envs.{env_name}.gym_wrapper")
    build_fn = getattr(module, f"build_{env_name}_env")
    env = build_fn()
    env._telemetry = telemetry
    return env
