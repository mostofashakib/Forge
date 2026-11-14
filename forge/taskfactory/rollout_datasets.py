"""Training rows built from one calibration run's graded attempts.

A frontier task shows the same model succeeding and failing at the same task,
which is the signal GRPO and preference training need. Every row's prompt is
the task's objective, the words the target model was given, and its
completion is the steps the model took. Rows are the shared dataset row
types, extended with what only calibration knows.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from forge.contracts.dataset_rows import (
    FailureEpisode,
    PreferencePair,
    RolloutRow,
    SftStep,
    exchange,
    step_exchanges,
    step_line,
    steps_text,
)
from forge.taskfactory.calibration import classify
from forge.taskfactory.schemas import SyntheticTask

FORMATS = ("grpo", "preference", "sft", "failures")


class CalibrationPair(PreferencePair):
    rejected_failed_checks: list[str]


class CalibrationSftStep(SftStep):
    env_name: str
    task_name: str


class CalibrationFailure(FailureEpisode):
    prompt: str
    termination: str
    error: str | None
    checks: list[dict]
    failed_checks: list[str]
    labels: dict


@dataclass(frozen=True)
class Calibrated:
    """A task and its stored attempts, as `calibration_registry.task_attempts` returns them."""

    task: SyntheticTask
    attempts: Sequence[dict]

    @property
    def on_frontier(self) -> bool:
        return classify(sum(a["passed"] for a in self.attempts) / len(self.attempts)) == "frontier"

    @property
    def model_attempts(self) -> list[dict]:
        """Attempts the model played. A failed seed never reached the model, so it teaches nothing."""
        return [a for a in self.attempts if a["termination"] != "seed_error"]


@dataclass(frozen=True)
class _Run:
    env_name: str
    run_id: str
    target_model: str

    def episode_id(self, task: SyntheticTask, attempt: dict) -> str:
        return f"{self.run_id}/{task.id}/{attempt['attempt_index']}"


def dataset_rows(
    fmt: str, calibrated: Sequence[Calibrated], *, env_name: str, run_id: str, target_model: str
) -> list[dict]:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, got {fmt!r}")
    run = _Run(env_name, run_id, target_model)
    if fmt == "failures":
        return [_failure(run, item.task, a) for item in calibrated for a in item.attempts if not a["passed"]]
    frontier = [item for item in calibrated if item.on_frontier]
    if fmt == "grpo":
        return [_rollout(run, item.task, a) for item in frontier for a in item.model_attempts]
    if fmt == "sft":
        return [
            row for item in frontier for a in item.model_attempts if a["passed"]
            for row in _demonstrations(run, item.task, a)
        ]
    return [pair for item in frontier for pair in _pairs(run, item)]


def _rollout(run: _Run, task: SyntheticTask, attempt: dict) -> dict:
    reward = 1.0 if attempt["passed"] else 0.0
    steps = len(attempt["steps"])
    return RolloutRow(
        episode_id=run.episode_id(task, attempt),
        env_name=run.env_name, task_name=task.id, prompt=task.objective, completion=_completion(attempt),
        total_reward=reward, passed=attempt["passed"], outcome="success" if attempt["passed"] else "failure",
        steps=steps, error=attempt["error"] or None, behavior_model=run.target_model,
        termination_reason=attempt["termination"], verification_results=attempt["checks"],
        reward_breakdown={"total_reward": reward, "components": [{"name": "task_checks", "value": reward}]},
        agent_id=run.target_model, total_steps=steps,
    ).model_dump()


def _pairs(run: _Run, item: Calibrated) -> list[dict]:
    """Each failing attempt against a passing one, taking the passes in turn. Both sides are real attempts."""
    passes = [a for a in item.model_attempts if a["passed"]]
    fails = [a for a in item.model_attempts if not a["passed"]]
    if not passes or not fails:
        return []
    return [
        CalibrationPair(
            env_name=run.env_name, task=item.task.id,
            chosen=exchange(item.task.objective, _completion(passes[i % len(passes)])),
            rejected=exchange(item.task.objective, _completion(failed)),
            chosen_reward=1.0, rejected_reward=0.0, chosen_passed=True, rejected_passed=False,
            rejected_failed_checks=failed["labels"]["failed_checks"],
        ).model_dump()
        for i, failed in enumerate(fails)
    ]


def _demonstrations(run: _Run, task: SyntheticTask, attempt: dict) -> list[dict]:
    lines = [step_line(turn["step"]) for turn in attempt["steps"]]
    return [
        CalibrationSftStep(
            messages=messages, episode_id=run.episode_id(task, attempt), step_index=index, total_reward=1.0,
            env_name=run.env_name, task_name=task.id,
        ).model_dump()
        for index, messages in enumerate(step_exchanges(task.objective, lines))
    ]


def _failure(run: _Run, task: SyntheticTask, attempt: dict) -> dict:
    return CalibrationFailure(
        episode_id=run.episode_id(task, attempt), env_name=run.env_name, task_name=task.id,
        total_reward=0.0, total_steps=len(attempt["steps"]),
        steps=[
            {"step_index": index, "command": step_line(turn["step"]), "action": turn["step"],
             "ok": turn["ok"], "output": turn["output"], "error": turn["error"]}
            for index, turn in enumerate(attempt["steps"])
        ],
        prompt=task.objective, termination=attempt["termination"], error=attempt["error"],
        checks=attempt["checks"], failed_checks=attempt["labels"]["failed_checks"], labels=attempt["labels"],
    ).model_dump()


def _completion(attempt: dict) -> str:
    return steps_text([turn["step"] for turn in attempt["steps"]])
