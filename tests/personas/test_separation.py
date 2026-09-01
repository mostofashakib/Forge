"""The persona stack's own boundaries.

Personas are the one place in Forge where an environment deliberately reaches
for a language model. That inversion is safe only while it stays confined to
the driver: the moment scheduling or roster resolution can call a model, the
determinism guarantee those two provide is gone, and nothing in the test suite
would notice.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PERSONAS = ROOT / "forge" / "personas"

# The deterministic half. Nothing here may reach a model, directly or through
# the driver layer that is allowed to.
DETERMINISTIC_MODULES = ["scheduler.py", "population.py", "guardrails.py", "archetypes.py"]


def imports_of(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_the_persona_package_is_where_this_test_thinks_it_is():
    assert (PERSONAS / "drivers.py").exists()


@pytest.mark.parametrize("module", DETERMINISTIC_MODULES)
def test_the_deterministic_half_cannot_reach_a_model(module):
    forbidden = ("forge.runtime.agents", "forge.personas.drivers", "anthropic", "openai")
    offenders = [m for m in imports_of(PERSONAS / module) if m.startswith(forbidden)]
    assert not offenders, (
        f"{module} imports {offenders} — scheduling and roster resolution must "
        "stay deterministic, so they may not reach the driver layer or a model"
    )


def test_the_driver_layer_is_the_only_one_that_may():
    """False-positive guard: the rule above must not be vacuous."""
    assert any(
        m.startswith("forge.runtime.agents") for m in imports_of(PERSONAS / "drivers.py")
    ), "drivers.py no longer reaches an agent — has the model path moved?"


def test_the_scheduler_never_reads_environment_state():
    """A scheduler that reads state makes persona timing agent-controllable."""
    source = (PERSONAS / "scheduler.py").read_text()
    assert "PersonaView" not in source
    assert ".state" not in source


def test_the_environment_core_does_not_import_an_agent_to_get_personas():
    """The model path stays lazy, inside the driver, not on the env's import."""
    forbidden = "forge.runtime.agents"
    for rel in ("forge/runtime/env.py", "forge/runtime/env_builder.py"):
        assert not any(
            m.startswith(forbidden) for m in imports_of(ROOT / rel)
        ), f"{rel} imports {forbidden}"


def test_the_contract_does_not_depend_on_its_implementation():
    offenders = [
        m
        for m in imports_of(ROOT / "forge" / "contracts" / "persona.py")
        if m.startswith(("forge.personas", "forge.runtime", "forge.envgen"))
    ]
    assert not offenders, offenders
