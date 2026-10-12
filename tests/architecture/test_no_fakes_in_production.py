"""Fakes belong to the tests. Production packages ship none."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAKE_NAME = re.compile(r"(Mock|Fake|Dummy|Stub)(?=[A-Z_]|$)")


def test_no_production_module_defines_a_fake_class():
    offenders = [
        f"{path.relative_to(ROOT)}:{node.lineno} {node.name}"
        for package in ("forge", "backend")
        for path in (ROOT / package).rglob("*.py")
        if "__pycache__" not in path.parts
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ClassDef) and FAKE_NAME.search(node.name)
    ]

    assert offenders == []
