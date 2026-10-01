"""Step 3b: run the golden solution k times and require an identical pass each time."""
from __future__ import annotations

from dataclasses import dataclass

from forge.taskfactory.runner import SeedError, TaskRunner
from forge.taskfactory.schemas import OUTCOME_KINDS, TaskDraft


@dataclass(frozen=True)
class PassKResult:
    passed: bool
    reason: str = ""
    fingerprint: str = ""


def run_pass_k(runner: TaskRunner, draft: TaskDraft, k: int) -> PassKResult:
    outcome_checks = [check for check in draft.checks if check.kind in OUTCOME_KINDS]
    fingerprints: list[str] = []
    for run in range(1, k + 1):
        try:
            with runner.session(draft) as session:
                before = session.evaluate(outcome_checks)
                if all(outcome.passed for outcome in before):
                    return PassKResult(False, f"run {run}: every outcome check already passes before the golden solution")
                for number, step in enumerate(draft.golden, start=1):
                    result = session.step(step)
                    if not result.ok:
                        return PassKResult(False, f"run {run}: golden step {number} ({step.tool}) failed: {result.error}")
                after = session.evaluate(draft.checks)
                failed = [
                    f"{check.description or check.kind}: {outcome.detail}"
                    for check, outcome in zip(draft.checks, after)
                    if not outcome.passed
                ]
                if failed:
                    return PassKResult(False, f"run {run}: checks failed after the golden solution: " + "; ".join(failed))
                fingerprints.append(session.fingerprint())
        except SeedError as exc:
            return PassKResult(False, f"run {run}: seed could not be applied: {exc}")
    if len(set(fingerprints)) > 1:
        return PassKResult(False, f"the final state differed across the {k} runs")
    return PassKResult(True, fingerprint=fingerprints[0])
