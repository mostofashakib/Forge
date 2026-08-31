from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable

from forge.contracts import RewardBreakdown, RewardComponent, Rubric

if TYPE_CHECKING:
    from forge.runtime.trajectory import Trajectory

__all__ = [
    "FunctionRubric",
    "RewardBreakdown",
    "RewardComponent",
    "RewardEngine",
    "TaskSuccessRubric",
]


class FunctionRubric(Rubric):
    """Adapts a plain `(state, trajectory, verifier_results, task)` callable."""

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        functools.update_wrapper(self, fn, updated=())

    @property
    def fn(self) -> Callable:
        """The plain callable this adapter wraps."""
        return self._fn

    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        return self._fn(state, trajectory, verifier_results, task)


class TaskSuccessRubric(Rubric):
    """The default rubric: 1.0 if any verifier passed, else 0.0."""

    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        passed = any(vr.passed for vr in verifier_results)
        value = 1.0 if passed else 0.0
        return RewardBreakdown(
            total_reward=value,
            components=[RewardComponent(name="task_success", value=value)],
        )


class RewardEngine:
    def __init__(self) -> None:
        self._task_rubrics: dict[str, Rubric] = {}
        self._default: Rubric | None = None

    def register(self, task_name: str, rubric: Rubric) -> None:
        if not isinstance(rubric, Rubric):
            raise TypeError(
                f"Rubric for {task_name!r} must be a Rubric, got "
                f"{type(rubric).__name__}. Wrap a plain function in FunctionRubric."
            )
        self._task_rubrics[task_name] = rubric

    def set_default(self, rubric: Rubric) -> None:
        if not isinstance(rubric, Rubric):
            raise TypeError(
                f"Default rubric must be a Rubric, got {type(rubric).__name__}. "
                f"Wrap a plain function in FunctionRubric."
            )
        self._default = rubric

    def compute(
        self,
        state: dict,
        trajectory: "Trajectory",
        verifier_results: list,
        task: dict | None = None,
    ) -> RewardBreakdown:
        task_name = task.get("name") if task else None
        rubric = self._task_rubrics.get(task_name) if task_name else None
        rubric = rubric or self._default or TaskSuccessRubric()
        return rubric.score(state, trajectory, verifier_results, task)
