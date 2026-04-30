from __future__ import annotations
import ast
import os
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_MODULES = frozenset([
    "requests", "httpx", "urllib", "socket", "http", "aiohttp",
])


@dataclass
class NetworkIsolationViolation:
    module: str
    import_line: str
    filename: str


def _is_forbidden(module_name: str) -> bool:
    root = module_name.split(".")[0]
    return root in FORBIDDEN_MODULES


def check_file(path: Path) -> list[NetworkIsolationViolation]:
    if os.environ.get("FORGE_DEV_NETWORK", "").lower() == "true":
        return []
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    violations: list[NetworkIsolationViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    violations.append(
                        NetworkIsolationViolation(
                            module=alias.name.split(".")[0],
                            import_line=f"import {alias.name}",
                            filename=str(path),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_forbidden(module):
                violations.append(
                    NetworkIsolationViolation(
                        module=module.split(".")[0],
                        import_line=f"from {module} import ...",
                        filename=str(path),
                    )
                )
    return violations


def check_generated_env(env_dir: Path) -> list[NetworkIsolationViolation]:
    if os.environ.get("FORGE_DEV_NETWORK", "").lower() == "true":
        return []
    all_violations: list[NetworkIsolationViolation] = []
    for py_file in env_dir.rglob("*.py"):
        all_violations.extend(check_file(py_file))
    return all_violations
