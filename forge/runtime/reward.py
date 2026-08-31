from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable

from forge.contracts import RewardBreakdown, RewardComponent, Rubric
from forge.runtime._signature import require_arity

if TYPE_CHECKING:
    from forge.runtime.trajectory import Trajectory

__all__ = [
    "FunctionRubric",
    "RewardBreakdown",
    "RewardComponent",
    "RewardEngine",
    "ObjectiveScoreRubric",
    "TaskSuccessRubric",
]


class FunctionRubric(Rubric):
    """Adapts a plain `(state, trajectory, verifier_results, task)` callable."""

    def __init__(self, fn: Callable) -> None:
        require_arity(
            fn, "FunctionRubric", ("state", "trajectory", "verifier_results", "task")
        )
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


class ObjectiveScoreRubric(Rubric):
    """Score a controller's objective verdict, with an optional progress floor."""

    def __init__(self, diff_floor: float = 0.0) -> None:
        self._diff_floor = diff_floor

    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        objective_score = max(
            (result.score for result in verifier_results), default=0.0
        )
        metadata = task.metadata if hasattr(task, "metadata") else (task or {})
        state_changed = bool(metadata.get("state_changed", False))
        value = max(objective_score, self._diff_floor) if state_changed else objective_score
        return RewardBreakdown(
            total_reward=value,
            components=[
                RewardComponent(name="objective_score", value=objective_score),
                RewardComponent(
                    name="state_change_floor",
                    value=max(0.0, value - objective_score),
                ),
            ],
        )


# One instance, not a fresh one per compute() call: the default rubric is
# stateless, so there is nothing to allocate per episode step.
_TASK_SUCCESS = TaskSuccessRubric()


class RewardEngine(Rubric):
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
        # Explicit `is None`, not truthiness: a Rubric is a user-supplied object
        # and may define __bool__ or __len__ (e.g. over its components), which
        # would otherwise silently fall through to the default.
        if rubric is None:
            rubric = self._default
        if rubric is None:
            rubric = _TASK_SUCCESS
        return rubric.score(state, trajectory, verifier_results, task)

    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        """Expose registry-backed scoring through the composed Rubric contract."""
        return self.compute(state, trajectory, list(verifier_results), task)
