"""Training rows built from one calibration run's graded attempts.

A frontier task shows the same model succeeding and failing at the same task,
which is the signal GRPO and preference training need. Every row's prompt is
the task's objective, the words the target model was given, and its
completion is the steps the model took.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from forge.contracts import RolloutRecord
from forge.taskfactory.calibration import classify
from forge.taskfactory.schemas import SyntheticTask

FORMATS = ("grpo", "preference", "sft", "failures")


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


def dataset_rows(
    fmt: str, calibrated: Sequence[Calibrated], *, env_name: str, run_id: str, target_model: str
) -> list[dict]:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, got {fmt!r}")
    if fmt == "failures":
        return [_failure(item.task, a) for item in calibrated for a in item.attempts if not a["passed"]]
    frontier = [item for item in calibrated if item.on_frontier]
    if fmt == "grpo":
        return [_rollout(item.task, a, env_name, run_id, target_model) for item in frontier for a in item.model_attempts]
    if fmt == "sft":
        return [_demonstration(item.task, a, env_name) for item in frontier for a in item.model_attempts if a["passed"]]
    return [pair for item in frontier for pair in _pairs(item, env_name)]


def _rollout(task: SyntheticTask, attempt: dict, env_name: str, run_id: str, target_model: str) -> dict:
    reward = 1.0 if attempt["passed"] else 0.0
    record = RolloutRecord(
        episode_id=f"{run_id}/{task.id}/{attempt['attempt_index']}",
        env_name=env_name, task_name=task.id, prompt=task.objective, completion=_completion(attempt),
        total_reward=reward, passed=attempt["passed"], outcome="success" if attempt["passed"] else "failure",
        steps=len(attempt["steps"]), error=attempt["error"] or None, behavior_model=target_model,
        termination_reason=attempt["termination"], verification_results=attempt["checks"],
        reward_breakdown={"total_reward": reward, "components": [{"name": "task_checks", "value": reward}]},
    )
    return record.model_dump()


def _pairs(item: Calibrated, env_name: str) -> list[dict]:
    """Each failing attempt against a passing one, taking the passes in turn. Both sides are real attempts."""
    passes = [a for a in item.model_attempts if a["passed"]]
    fails = [a for a in item.model_attempts if not a["passed"]]
    if not passes or not fails:
        return []
    return [
        {
            "env_name": env_name, "task": item.task.id,
            "chosen": _messages(item.task, passes[i % len(passes)]), "rejected": _messages(item.task, failed),
            "chosen_reward": 1.0, "rejected_reward": 0.0, "chosen_passed": True, "rejected_passed": False,
            "rejected_failed_checks": failed["labels"]["failed_checks"],
        }
        for i, failed in enumerate(fails)
    ]


def _demonstration(task: SyntheticTask, attempt: dict, env_name: str) -> dict:
    return {
        "messages": _messages(task, attempt), "env_name": env_name,
        "task_id": task.id, "attempt_index": attempt["attempt_index"], "reward": 1.0,
    }


def _failure(task: SyntheticTask, attempt: dict) -> dict:
    return {
        "task_id": task.id, "attempt_index": attempt["attempt_index"], "prompt": task.objective,
        "termination": attempt["termination"], "error": attempt["error"], "steps": attempt["steps"],
        "checks": attempt["checks"], "failed_checks": attempt["labels"]["failed_checks"], "labels": attempt["labels"],
    }


def _messages(task: SyntheticTask, attempt: dict) -> list[dict]:
    return [{"role": "user", "content": task.objective}, {"role": "assistant", "content": _completion(attempt)}]


def _completion(attempt: dict) -> str:
    return "\n".join(step_text(turn["step"]) for turn in attempt["steps"])


def step_text(step: dict) -> str:
    """One step as a line of a completion: a shell command, or a tool with its arguments."""
    args = step.get("args") or {}
    if step["tool"] == "shell":
        return f"$ {args.get('command', '')}"
    return f"{step['tool']} {json.dumps(args)}" if args else step["tool"]
