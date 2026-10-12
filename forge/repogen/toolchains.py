"""Per-language knowledge: how to spot a stack, install it, run its tests, and read results.

Results are read per test from a structured report (JUnit XML for pytest,
jest-style JSON for jest and vitest), so a baseline can keep the tests that
pass reliably instead of needing the whole suite green.
"""
from __future__ import annotations

import json
import tomllib
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Protocol

_NPM_PLACEHOLDER_TEST = "no test specified"
_PYTHON_EXTRAS = ("test", "tests", "testing", "dev")
_REQUIREMENT_FILES = (
    "requirements.txt", "requirements-dev.txt", "requirements-test.txt", "requirements_dev.txt",
    "dev-requirements.txt", "test-requirements.txt", "requirements/dev.txt", "requirements/test.txt",
)
_PIP = "pip install --no-cache-dir"


class Toolchain(Protocol):
    name: str
    base_image: str

    def detect(self, root: Path) -> bool: ...

    def install_commands(self, root: Path) -> list[str]: ...

    def test_command(self, report_path: str, root: Path) -> str: ...

    def parse_report(self, text: str) -> dict[str, bool]:
        """Test id to whether it passed. Skipped tests count as not passed."""


class PythonToolchain:
    name = "python"
    base_image = "python:3.12-slim"

    def detect(self, root: Path) -> bool:
        return any((root / name).exists() for name in ("pyproject.toml", "setup.py", "setup.cfg")) or any(
            (root / name).exists() for name in _REQUIREMENT_FILES
        )

    def install_commands(self, root: Path) -> list[str]:
        commands: list[str] = []
        if any((root / name).exists() for name in ("pyproject.toml", "setup.py", "setup.cfg")):
            extra = _first_python_extra(root)
            commands.append(f'{_PIP} -e ".[{extra}]"' if extra else f"{_PIP} -e .")
        commands += [f"{_PIP} -r {name}" for name in _REQUIREMENT_FILES if (root / name).exists()]
        commands.append(f"{_PIP} pytest")
        return commands

    def test_command(self, report_path: str, root: Path) -> str:
        return f"python -m pytest -q -p no:cacheprovider --junitxml={report_path}"

    def parse_report(self, text: str) -> dict[str, bool]:
        try:
            tree = ElementTree.fromstring(text)
        except ElementTree.ParseError as exc:
            raise ValueError(f"unreadable JUnit report: {exc}") from exc
        return {
            f"{case.get('classname', '')}::{case.get('name', '')}": not any(
                child.tag in {"failure", "error", "skipped"} for child in case
            )
            for case in tree.iter("testcase")
        }


class NodeToolchain:
    name = "node"
    base_image = "node:20-slim"

    def detect(self, root: Path) -> bool:
        package = _package_json(root)
        if package is None:
            return False
        script = (package.get("scripts") or {}).get("test", "")
        return bool(script) and _NPM_PLACEHOLDER_TEST not in script and _test_runner(package) is not None

    def install_commands(self, root: Path) -> list[str]:
        if (root / "package-lock.json").exists():
            return ["npm ci"]
        if (root / "pnpm-lock.yaml").exists():
            return ["corepack enable && pnpm install --frozen-lockfile"]
        if (root / "yarn.lock").exists():
            return ["corepack enable && yarn install --frozen-lockfile"]
        return ["npm install"]

    def test_command(self, report_path: str, root: Path) -> str:
        if _test_runner(_package_json(root) or {}) == "vitest":
            return f"npx vitest run --reporter=json --outputFile={report_path}"
        return f"npx jest --ci --json --outputFile={report_path}"

    def parse_report(self, text: str) -> dict[str, bool]:
        try:
            report = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"unreadable jest/vitest report: {exc}") from exc
        results: dict[str, bool] = {}
        for suite in report.get("testResults", []):
            path = str(suite.get("name", "")).removeprefix("/repo/")
            for assertion in suite.get("assertionResults", []):
                results[f"{path}::{assertion.get('fullName', '')}"] = assertion.get("status") == "passed"
        return results


TOOLCHAINS: tuple[Toolchain, ...] = (PythonToolchain(), NodeToolchain())
_PREFERRED = {"Python": "python", "TypeScript": "node", "JavaScript": "node"}


def detect_toolchain(root: Path, language: str | None) -> Toolchain | None:
    """The toolchain for `root`, preferring the one matching GitHub's language when both fit."""
    preferred = _PREFERRED.get(language or "")
    ordered = sorted(TOOLCHAINS, key=lambda toolchain: toolchain.name != preferred)
    return next((toolchain for toolchain in ordered if toolchain.detect(root)), None)


def _first_python_extra(root: Path) -> str | None:
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text()).get("project", {})
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return None
    extras = project.get("optional-dependencies") or {}
    return next((name for name in _PYTHON_EXTRAS if name in extras), None)


def _package_json(root: Path) -> dict | None:
    try:
        package = json.loads((root / "package.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return package if isinstance(package, dict) else None


def _test_runner(package: dict) -> str | None:
    deps = {**(package.get("dependencies") or {}), **(package.get("devDependencies") or {})}
    return next((runner for runner in ("vitest", "jest") if runner in deps), None)
