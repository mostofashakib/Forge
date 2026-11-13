"""One model attempt at a stored task, graded by the task's own checks.

A failed step is shown to the model and the attempt goes on, as it would for
an agent exploring a shell or an app. The attempt ends when the model submits,
runs out of steps, or makes a move the environment cannot run. Only the last
of these is a policy error. A provider outage is not the model's fault, so it
propagates instead of being graded.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from forge.runtime.errors import AgentError
from forge.taskfactory.policy import Policy, PolicyError, Submit, Turn
from forge.taskfactory.runner import SeedError, TaskRunner
from forge.taskfactory.schemas import SyntheticTask
from forge.taskfactory.state_checks import CheckOutcome

Termination = Literal["submitted", "budget", "policy_error", "seed_error"]
# The model chose something that cannot run: no tool call, a tool the
# environment lacks, or arguments that are not JSON.
_MODEL_FAULTS = (AgentError, PolicyError, json.JSONDecodeError)


@dataclass(frozen=True)
class Attempt:
    turns: tuple[Turn, ...]
    checks: tuple[CheckOutcome, ...]
    termination: Termination
    error: str = ""

    @property
    def passed(self) -> bool:
        graded = self.termination in ("submitted", "budget")
        return graded and bool(self.checks) and all(outcome.passed for outcome in self.checks)


def run_attempt(runner: TaskRunner, task: SyntheticTask, policy: Policy) -> Attempt:
    try:
        with runner.session(task) as session:
            turns: list[Turn] = []
            termination: Termination = "budget"
            error = ""
            while len(turns) < task.step_budget:
                try:
                    choice = policy.next_step(turns)
                except _MODEL_FAULTS as exc:
                    termination, error = "policy_error", f"{type(exc).__name__}: {exc}"
                    break
                if isinstance(choice, Submit):
                    termination = "submitted"
                    break
                turns.append(Turn(choice, session.step(choice)))
            checks = session.evaluate(task.checks)
    except SeedError as exc:
        return Attempt(turns=(), checks=(), termination="seed_error", error=str(exc))
    return Attempt(turns=tuple(turns), checks=tuple(checks), termination=termination, error=error)


def attempt_labels(attempt: Attempt, task: SyntheticTask) -> dict:
    """Why an attempt went the way it did, from what was recorded. No model is asked."""
    golden_tools = {step.tool for step in task.golden}
    return {
        "passed": attempt.passed,
        "termination": attempt.termination,
        "failed_checks": [
            _check_name(check)
            for check, outcome in zip(task.checks, attempt.checks)
            if not outcome.passed
        ],
        "step_errors": sum(not turn.result.ok for turn in attempt.turns),
        "off_golden_tools": sorted({turn.step.tool for turn in attempt.turns} - golden_tools),
    }


def summarize_failures(labels: Sequence[dict]) -> dict | None:
    """Why a task's failed attempts failed, counted across their labels. None when every attempt passed."""
    if not labels:
        raise ValueError("a failure summary needs at least one attempt")
    failed = [label for label in labels if not label["passed"]]
    if not failed:
        return None
    terminations = Counter(label["termination"] for label in failed)
    failed_checks = Counter(name for label in failed for name in label["failed_checks"])
    causes = failed_checks + Counter({
        _ENDINGS[name]: count for name, count in terminations.items() if name in _ENDINGS
    })
    cause, count = causes.most_common(1)[0]
    endings = ", ".join(f"{name} {n}" for name, n in terminations.most_common())
    return {
        "attempts": len(labels),
        "failed": len(failed),
        "terminations": dict(terminations),
        "failed_checks": dict(failed_checks),
        "text": (
            f"{len(failed)} of {len(labels)} attempts failed. "
            f"Most common cause: {cause} ({count}). Ended by: {endings}."
        ),
    }


_ENDINGS = {
    "policy_error": "the model made a move the environment could not run",
    "seed_error": "the task's seed could not be applied",
}


# Characters of a check's name in labels and summaries.
_NAME_LIMIT = 120


def _check_name(check) -> str:
    """The check's description, or its kind and what it looks at, so different checks never share a name."""
    if check.description:
        return _cut(check.description)
    subject = check.command or check.selector or check.path or check.collection or ", ".join(check.tools or [])
    return _cut(f"{check.kind}: {subject}" if subject else check.kind)


def _cut(text: str) -> str:
    return text if len(text) <= _NAME_LIMIT else text[:_NAME_LIMIT - 1] + "…"
