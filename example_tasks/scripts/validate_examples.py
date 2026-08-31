#!/usr/bin/env python3
"""Dependency-free validation for the Forge RL example tasks."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "slack_task_1": "forge/slack-task-1",
    "slack_task_2": "forge/slack-task-2",
    "task_manager_task_1": "forge/task-manager-task-1",
}
REQUIRED_TASK_FILES = (
    "README.md",
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
    "solution/solve.sh",
    "tests/Dockerfile",
    "tests/check.py",
    "tests/docker-compose.yaml",
    "tests/test.sh",
)


def validate_task(task_dir_name: str, expected_name: str) -> list[str]:
    errors: list[str] = []
    task_dir = ROOT / task_dir_name

    for relative_path in REQUIRED_TASK_FILES:
        path = task_dir / relative_path
        if not path.is_file():
            errors.append(f"{task_dir_name}: missing {relative_path}")

    config_path = task_dir / "task.toml"
    if config_path.is_file():
        try:
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{task_dir_name}: invalid task.toml: {exc}")
        else:
            actual_name = config.get("task", {}).get("name")
            if actual_name != expected_name:
                errors.append(
                    f"{task_dir_name}: task.name must be {expected_name!r}, got {actual_name!r}"
                )
            if config.get("schema_version") != "1.3":
                errors.append(f"{task_dir_name}: schema_version must be '1.3'")
            artifacts = set(config.get("artifacts", []))
            expected_artifacts = {
                "/logs/agent/trajectory.json",
                "/logs/agent/trajectory.txt",
            }
            if artifacts != expected_artifacts:
                errors.append(f"{task_dir_name}: unexpected artifact declaration")

    instruction_path = task_dir / "instruction.md"
    if instruction_path.is_file() and not instruction_path.read_text(encoding="utf-8").strip():
        errors.append(f"{task_dir_name}: instruction.md is empty")

    readme_path = task_dir / "README.md"
    if readme_path.is_file():
        readme = readme_path.read_text(encoding="utf-8")
        if "./run.sh" not in readme:
            errors.append(f"{task_dir_name}: README must document the root runner")
        if "scripts/build_harbor_images.sh" in readme:
            errors.append(f"{task_dir_name}: README references a removed build script")

    for relative_path in ("solution/solve.sh", "tests/test.sh"):
        script_path = task_dir / relative_path
        if script_path.is_file() and not script_path.read_text(encoding="utf-8").startswith("#!"):
            errors.append(f"{task_dir_name}: {relative_path} needs a shebang")

    return errors


def validate_python_syntax() -> list[str]:
    errors: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".venv", "jobs"} for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: Python syntax error: {exc}")
    return errors


def main() -> int:
    errors: list[str] = []
    for task_dir_name, expected_name in TASKS.items():
        errors.extend(validate_task(task_dir_name, expected_name))
    errors.extend(validate_python_syntax())

    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Validated {len(TASKS)} RL tasks and all Python source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
