"""RewardKit criteria for Harbor task verifiers."""

from __future__ import annotations

import importlib
import json
import re
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

import rewardkit as rk


def load_symbol(import_path: str, trajectory_path: str | Path | None = None):
    module_name, symbol_name = import_path.split(":", maxsplit=1)
    if module_name == "check":
        import sys

        if Path("/tests").is_dir():
            if "/tests" not in sys.path:
                sys.path.insert(0, "/tests")
        elif trajectory_path is not None:
            current_path = Path(trajectory_path).resolve().parent
            for _ in range(5):
                tests_dir = current_path / "tests"
                if (tests_dir / "check.py").is_file():
                    if str(tests_dir) not in sys.path:
                        sys.path.insert(0, str(tests_dir))
                    break
                if (current_path / "check.py").is_file():
                    if str(current_path) not in sys.path:
                        sys.path.insert(0, str(current_path))
                    break
                current_path = current_path.parent
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


# Workspace database files the verifier reads directly. SQLite is the single
# source of truth: when the workspace exposes the service database, the final
# state comes from it, never from the agent-written trajectory snapshot.
_WORKSPACE_DATABASES: dict[str, str] = {
    "slack.db": "fleet.environments.slack.sqlite_service",
    "task_manager.db": "fleet.environments.task_manager.sqlite_service",
}


def find_workspace_database(workspace: str | Path | None) -> Path | None:
    if workspace is None:
        return None
    for filename in _WORKSPACE_DATABASES:
        candidate = Path(workspace) / filename
        if candidate.is_file():
            return candidate
    return None


def load_final_state_from_database(db_path: Path) -> dict[str, Any]:
    module = importlib.import_module(_WORKSPACE_DATABASES[db_path.name])
    return module.export_state(db_path)


def evaluate_verifier(trajectory_path: str | Path, spec: str, workspace: str | Path | None = None) -> dict[str, Any]:
    try:
        trajectory = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
        db_path = find_workspace_database(workspace)
        if db_path is not None:
            final_state = load_final_state_from_database(db_path)
            # Every final-state consumer must see the database truth, including
            # trajectory checks that compare snapshots.
            trajectory.setdefault("extra", {})["final_state_snapshot"] = final_state
        else:
            final_state = trajectory.get("extra", {}).get("final_state_snapshot", {})
        verifier_factory = load_symbol(spec, trajectory_path)
        results = verifier_factory().verify(final_state, trajectory)
        passed = all(result.passed for result in results)
        return {
            "passed": passed,
            "score": 1 if passed else 0,
            "max_score": 1,
            "reward": 1 if passed else 0,
            "results": [
                {
                    "index": index,
                    "layer": result.layer,
                    "passed": result.passed,
                    "score": result.score,
                    "message": result.message,
                    "details": result.details,
                }
                for index, result in enumerate(results)
            ],
        }
    except Exception as exc:
        return {
            "passed": False,
            "score": 0,
            "max_score": 1,
            "reward": 0,
            "results": [
                {
                    "layer": "verifier_exception",
                    "passed": False,
                    "score": 0,
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {"cause": repr(exc.__cause__) if exc.__cause__ else None},
                }
            ],
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "cause": repr(exc.__cause__) if exc.__cause__ else None,
                "traceback": traceback.format_exc(),
            },
        }


def write_report(report: dict[str, Any], report_path: str | Path, reward_path: str | Path | None = None) -> None:
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    if reward_path is not None:
        reward_target = Path(reward_path)
        reward_target.parent.mkdir(parents=True, exist_ok=True)
        reward_target.write_text(f"{int(report['reward'])}\n", encoding="utf-8")


@lru_cache(maxsize=128)
def _cached_report(
    trajectory_path: str,
    spec: str,
    modified_ns: int,
    size: int,
    workspace: str | None,
    db_modified_ns: int,
    db_size: int,
) -> dict[str, Any]:
    del modified_ns, size, db_modified_ns, db_size
    return evaluate_verifier(trajectory_path, spec, workspace)


def _report(trajectory_path: str, spec: str, workspace: str | Path | None = None) -> dict[str, Any]:
    path = Path(trajectory_path)
    try:
        stat = path.stat()
    except OSError:
        return evaluate_verifier(trajectory_path, spec, workspace)
    db_path = find_workspace_database(workspace)
    db_stat = db_path.stat() if db_path is not None else None
    return _cached_report(
        trajectory_path,
        spec,
        stat.st_mtime_ns,
        stat.st_size,
        str(workspace) if workspace is not None else None,
        db_stat.st_mtime_ns if db_stat else 0,
        db_stat.st_size if db_stat else 0,
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:80] or "check"


@rk.criterion(description="all deterministic verifier checks pass for {task_name}")
def harbor_all_checks_pass(
    workspace: Path,
    spec: str,
    task_name: str,
    trajectory_path: str = "/logs/agent/trajectory.json",
    report_path: str = "/logs/verifier/report.json",
) -> bool:
    report = _report(str(trajectory_path), spec, workspace)
    write_report(report, report_path)
    return bool(report["passed"])


@rk.criterion(description="{check_label}")
def harbor_verifier_check_passes(
    workspace: Path,
    spec: str,
    result_index: int,
    check_label: str,
    trajectory_path: str = "/logs/agent/trajectory.json",
) -> bool:
    del check_label
    report = _report(str(trajectory_path), spec, workspace)
    results = report.get("results", [])
    if result_index < 0 or result_index >= len(results):
        return False
    return bool(results[result_index].get("passed", False))


def register_harbor_verifier(
    spec: str,
    task_name: str,
    check_labels: list[str],
    trajectory_path: str = "/logs/agent/trajectory.json",
    report_path: str = "/logs/verifier/report.json",
) -> None:
    rk.harbor_all_checks_pass(
        spec,
        task_name,
        trajectory_path=trajectory_path,
        report_path=report_path,
        name=f"{task_name}.all_checks_pass",
    )
    for result_index, check_label in enumerate(check_labels):
        rk.harbor_verifier_check_passes(
            spec,
            result_index,
            check_label,
            trajectory_path=trajectory_path,
            weight=0,
            name=f"{task_name}.{result_index:02d}_{_slug(check_label)}",
        )
