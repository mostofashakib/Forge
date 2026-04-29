from __future__ import annotations
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    passed: bool
    output: str
    total_tests: int
    failed_tests: int


class ValidationRunner:
    def __init__(self, generated_envs_root: Path) -> None:
        self._root = generated_envs_root

    def run(self, pkg_dir: Path) -> ValidationResult:
        tests_dir = pkg_dir / "tests"
        if not tests_dir.exists():
            return ValidationResult(passed=False, output="No tests directory found", total_tests=0, failed_tests=0)

        env_copy = {
            **_current_env(),
            "PYTHONPATH": f"{self._root.parent}:{self._root}{_sep()}{_current_pythonpath()}",
        }
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(tests_dir), "-v", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            env=env_copy,
        )
        output = result.stdout + result.stderr
        total, failed = _parse_counts(output)
        return ValidationResult(
            passed=result.returncode == 0,
            output=output,
            total_tests=total,
            failed_tests=failed,
        )


def _current_env() -> dict:
    import os
    return dict(os.environ)


def _current_pythonpath() -> str:
    import os
    return os.environ.get("PYTHONPATH", "")


def _sep() -> str:
    import os
    return os.pathsep


def _parse_counts(output: str) -> tuple[int, int]:
    import re
    m = re.search(r"(\d+) passed", output)
    total = int(m.group(1)) if m else 0
    m2 = re.search(r"(\d+) failed", output)
    failed = int(m2.group(1)) if m2 else 0
    return total + failed, failed
