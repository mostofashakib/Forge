from __future__ import annotations
import subprocess
import sys
import json
import tempfile
import os
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

        forge_root = Path(__file__).resolve().parents[2]
        system_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
        read_roots = system_roots | {pkg_dir.resolve(), forge_root / "forge"}
        with tempfile.TemporaryDirectory(prefix="forge-validation-") as tmpdir:
            env_copy = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": f"{self._root.parent}:{self._root}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "FORGE_ENV": "test",
                "TMPDIR": tmpdir,
                "FORGE_VALIDATION_READ_ROOTS": json.dumps(
                    [str(path) for path in sorted(read_roots, key=str)] + [tmpdir]
                ),
                "FORGE_VALIDATION_WRITE_ROOTS": json.dumps(
                    [str(pkg_dir.resolve()), tmpdir]
                ),
            }
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "forge.compiler.sandbox_runner",
                    str(tests_dir),
                    "-v",
                    "--tb=short",
                    "--no-header",
                    "-p",
                    "no:cacheprovider",
                ],
                capture_output=True,
                text=True,
                env=env_copy,
                cwd=pkg_dir,
                timeout=60,
            )
        output = result.stdout + result.stderr
        total, failed = _parse_counts(output)
        return ValidationResult(
            passed=result.returncode == 0,
            output=output,
            total_tests=total,
            failed_tests=failed,
        )


def _parse_counts(output: str) -> tuple[int, int]:
    import re
    m = re.search(r"(\d+) passed", output)
    total = int(m.group(1)) if m else 0
    m2 = re.search(r"(\d+) failed", output)
    failed = int(m2.group(1)) if m2 else 0
    return total + failed, failed
