"""Only forge/credentials.py reads credentials. SDK clients receive keys explicitly."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from forge.credentials import CREDENTIALS

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "forge" / "credentials.py"
SDK_CLIENTS = {("anthropic", "Anthropic"), ("openai", "OpenAI"), ("genai", "Client")}


def _sources() -> list[Path]:
    return [
        path
        for package in ("forge", "backend")
        for path in (ROOT / package).rglob("*.py")
        if path != REGISTRY and "__pycache__" not in path.parts
    ]


def test_no_module_outside_the_registry_names_a_credential_variable():
    names = [name for credential in CREDENTIALS for name in credential.env_vars]
    literal = re.compile(r"""["'](%s)["']""" % "|".join(names))

    offenders = [
        f"{path.relative_to(ROOT)}: {match.group(1)}"
        for path in _sources()
        for match in literal.finditer(path.read_text())
    ]

    assert offenders == []


def test_every_sdk_client_is_given_its_key_explicitly():
    offenders = []
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            owner = node.func.value
            if not isinstance(owner, ast.Name) or (owner.id, node.func.attr) not in SDK_CLIENTS:
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            if "api_key" not in keywords:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert offenders == []
