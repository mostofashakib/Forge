"""forge/contracts/ must not depend on the packages that depend on it."""
from __future__ import annotations

import ast
import pathlib

import pytest

CONTRACTS = pathlib.Path(__file__).resolve().parents[2] / "forge" / "contracts"
FORBIDDEN = ("forge.runtime", "forge.envgen", "forge.extraction", "backend")


def _runtime_imports(source: str) -> list[str]:
    """Module names imported at runtime, ignoring `if TYPE_CHECKING:` blocks."""
    tree = ast.parse(source)
    type_checking_blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and (
            (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
            or (isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")
        )
    ]
    guarded = {id(child) for block in type_checking_blocks for child in ast.walk(block)}

    names: list[str] = []
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize(
    "path", sorted(CONTRACTS.rglob("*.py")), ids=lambda p: p.name
)
def test_contracts_module_has_no_forbidden_runtime_import(path):
    for name in _runtime_imports(path.read_text()):
        assert not name.startswith(FORBIDDEN), (
            f"{path.name} imports {name!r} at runtime; contracts/ must not "
            f"depend on {FORBIDDEN}. Move it under `if TYPE_CHECKING:`."
        )


def test_type_checking_imports_are_allowed():
    """False-positive guard: the guard must not reject a TYPE_CHECKING import."""
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from forge.runtime.context import RuntimeContext\n"
    )
    assert _runtime_imports(source) == ["typing"]


def test_plain_forbidden_import_is_detected():
    """Negative: an unguarded forbidden import must be caught."""
    source = "from forge.runtime.context import RuntimeContext\n"
    assert any(n.startswith(FORBIDDEN) for n in _runtime_imports(source))
