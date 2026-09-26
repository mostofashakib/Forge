from __future__ import annotations
import importlib
import sys

from forge.runtime.determinism import run_determinism_check
from forge.runtime.network_isolation import check_generated_env
from forge.settings import determinism_enabled, generated_envs_root


def load_forge_env(env_name: str, telemetry):
    """Dynamically import a generated ForgeEnv, verify determinism, inject telemetry."""
    envs_root = generated_envs_root()
    parent = str(envs_root.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    # The agent's tools are this package's transitions, and it runs inside the
    # worker, so a network import would give the agent the internet. Checked
    # before import, and never skipped by the determinism ablation.
    violations = check_generated_env(envs_root / env_name)
    if violations:
        details = ", ".join(f"{v.filename}: {v.import_line}" for v in violations)
        raise RuntimeError(f"Environment {env_name!r} violates network policy: {details}")
    module = importlib.import_module(f"generated_envs.{env_name}.gym_wrapper")
    build_fn = getattr(module, f"build_{env_name}_env")
    env = build_fn()
    # Verify before telemetry injection so check steps are never recorded.
    if determinism_enabled():
        run_determinism_check(env)
    env._telemetry = telemetry
    return env
