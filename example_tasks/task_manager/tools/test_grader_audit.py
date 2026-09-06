#!/usr/bin/env python3
"""The audit tool's own tests. Not part of the graded suite."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import grader_audit  # noqa: E402
from workspace import ORACLE_STEPS, Workspace, run_steps  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def test_it_finds_the_export_where_harbor_files_it() -> None:
    with tempfile.TemporaryDirectory() as directory:
        trial = Path(directory)
        nested = trial / "artifacts" / "var" / "lib" / "tasks"
        nested.mkdir(parents=True)
        (nested / "state-export.json").write_text("{}", encoding="utf-8")
        check("the collected artifact path is found",
              grader_audit.find_state_export(trial) == nested / "state-export.json")

    with tempfile.TemporaryDirectory() as directory:
        trial = Path(directory)
        (trial / "state-export.json").write_text("{}", encoding="utf-8")
        check("a flat export is found too",
              grader_audit.find_state_export(trial) == trial / "state-export.json")

    with tempfile.TemporaryDirectory() as directory:
        check("a trial with no export reports none",
              grader_audit.find_state_export(Path(directory)) is None)


def test_it_reads_both_export_shapes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.json"
        path.write_text(json.dumps({"result": {"tasks": []}}), encoding="utf-8")
        check("a socket envelope is unwrapped", grader_audit.load_state(path) == {"tasks": []})
        path.write_text(json.dumps({"tasks": []}), encoding="utf-8")
        check("a bare export is read as-is", grader_audit.load_state(path) == {"tasks": []})


def test_offline_reports_what_the_rules_did() -> None:
    with Workspace() as workspace:
        run_steps(workspace, ORACLE_STEPS)
        state = workspace.state()
    with tempfile.TemporaryDirectory() as directory:
        export = Path(directory) / "state-export.json"
        export.write_text(json.dumps(state), encoding="utf-8")
        audit = grader_audit.audit_trial(export, offline=True, model="unused", api_key="")
    check("the oracle's reward is reproduced", abs(audit.reward - 1.0) < 1e-9, str(audit.reward))
    check("every rule is reported", len(audit.verdicts) > 10, str(len(audit.verdicts)))
    check("no rule failed", not [v for v in audit.verdicts if not v.passed])
    check("and no model was consulted", audit.model_error is None and audit.disagreements == ())


def test_an_unreadable_model_answer_is_not_a_clean_bill_of_health() -> None:
    original = grader_audit.urllib.request.urlopen

    class _Response:
        def __init__(self, text: str) -> None:
            self._text = text

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": self._text}}]}
            ).encode("utf-8")

    try:
        grader_audit.urllib.request.urlopen = lambda *a, **k: _Response("I think it was fine.")
        found, error = grader_audit.ask_model((), "m", "k")
        check("prose instead of JSON is an error", error is not None and "unparseable" in error,
              str(error))
        check("and reports no findings it did not have", found == [])

        grader_audit.urllib.request.urlopen = lambda *a, **k: _Response('[{"name":"x","why":"y"}]')
        found, error = grader_audit.ask_model((), "m", "k")
        check("a well-formed answer is read", error is None and len(found) == 1, str(found))
    finally:
        grader_audit.urllib.request.urlopen = original


def test_the_grader_cannot_depend_on_its_auditor() -> None:
    root = Path(__file__).resolve().parents[1] / "verifiers"
    offenders = [
        path.name for path in root.rglob("*.py")
        if "grader_audit" in path.read_text(encoding="utf-8")
    ]
    check("no verifier module imports the audit tool", not offenders, str(offenders))


def main() -> int:
    print(__doc__)
    for test in (
        test_it_finds_the_export_where_harbor_files_it,
        test_it_reads_both_export_shapes,
        test_offline_reports_what_the_rules_did,
        test_an_unreadable_model_answer_is_not_a_clean_bill_of_health,
        test_the_grader_cannot_depend_on_its_auditor,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all grader-audit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
