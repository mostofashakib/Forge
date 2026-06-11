from __future__ import annotations
import importlib
import os
import sys
from pathlib import Path

from forge.runtime.determinism import run_determinism_check


def load_forge_env(env_name: str, telemetry):
    """Dynamically import a generated ForgeEnv, verify determinism, inject telemetry."""
    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
    parent = str(envs_root.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"generated_envs.{env_name}.gym_wrapper")
    build_fn = getattr(module, f"build_{env_name}_env")
    env = build_fn()
    # Verify before telemetry injection so check steps are never recorded.
    # Raises DeterminismError if two identically-seeded rollouts diverge.
    if os.environ.get("FORGE_SKIP_DETERMINISM_CHECK") != "1":
        run_determinism_check(env)
    env._telemetry = telemetry
    return env
