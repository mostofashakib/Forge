# tests/architecture/test_separation_of_concerns.py
"""Enforce the boundaries between environment, agent, verifier, and training code.

Each concern may only know about the layers below it:
  - agents see only observations and action types — never env internals,
    verifiers, rewards, or training code
  - the environment core never depends on agents or training
  - verifiers judge state and trajectories — never call agents or env internals
  - training/export consumes recorded episodes — never drives agents or envs
"""
from __future__ import annotations
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

ENV_CORE = [
    "forge/runtime/env.py",
    "forge/runtime/transition.py",
    "forge/runtime/state.py",
    "forge/runtime/context.py",
    "forge/runtime/snapshot.py",
    "forge/runtime/env_builder.py",
    "forge/runtime/interaction.py",
]
AGENTS = sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "forge/runtime/agents").glob("*.py")
)
VERIFIERS = [
    "forge/runtime/verifier.py",
    "forge/runtime/verification.py",
    "forge/runtime/layered_verifier.py",
    "forge/runtime/reward_hacking.py",
]
TRAINING = sorted(
    str(p.relative_to(ROOT))
    for d in ("backend/app/services/export_writers", "forge/benchmark")
    for p in (ROOT / d).glob("*.py")
)


def imports_of(rel_path: str) -> set[str]:
    tree = ast.parse((ROOT / rel_path).read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def assert_no_imports(files: list[str], forbidden_prefixes: tuple[str, ...], why: str) -> None:
    violations = []
    for rel_path in files:
        for module in imports_of(rel_path):
            if module.startswith(forbidden_prefixes):
                violations.append(f"{rel_path} imports {module}")
    assert not violations, f"{why}:\n" + "\n".join(violations)


def test_layers_are_populated():
    assert AGENTS and TRAINING  # glob found the packages


@pytest.mark.parametrize("rel_path", AGENTS)
def test_agents_only_depend_on_the_agent_package(rel_path):
    # forge.runtime.errors is the shared error vocabulary, importable from any layer.
    for module in imports_of(rel_path):
        if module.startswith(("forge.", "backend.")):
            assert module.startswith(("forge.runtime.agents", "forge.runtime.errors")), (
                f"{rel_path} imports {module} — agents must interact with environments "
                "only through observations and action types"
            )


def test_environment_core_does_not_know_agents_or_training():
    assert_no_imports(
        ENV_CORE,
        ("forge.runtime.agents", "forge.benchmark", "backend."),
        "environment core must not depend on agents or training",
    )


def test_verifiers_do_not_know_agents_env_or_training():
    assert_no_imports(
        VERIFIERS,
        (
            "forge.runtime.agents",
            "forge.runtime.env",
            "forge.runtime.env_builder",
            "forge.benchmark",
            "backend.",
        ),
        "verifiers judge state and trajectories only",
    )


def test_training_does_not_drive_agents_or_envs():
    assert_no_imports(
        TRAINING,
        ("forge.runtime.agents", "forge.runtime.env"),
        "training/export must consume recorded episodes, not drive agents or envs",
    )
