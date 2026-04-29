from __future__ import annotations
from typing import Callable
from pydantic import BaseModel


class RewardComponent(BaseModel):
    name: str
    value: float


class RewardBreakdown(BaseModel):
    total_reward: float
    components: list[RewardComponent]


class RewardEngine:
    def __init__(self) -> None:
        self._task_fns: dict[str, Callable] = {}
        self._default_fn: Callable | None = None

    def register(self, task_name: str, fn: Callable) -> None:
        self._task_fns[task_name] = fn

    def set_default(self, fn: Callable) -> None:
        self._default_fn = fn

    def compute(
        self,
        state: dict,
        trajectory: "Trajectory",
        verifier_results: list,
        task: dict | None = None,
    ) -> RewardBreakdown:
        task_name = task.get("name") if task else None
        fn = self._task_fns.get(task_name) if task_name else None
        fn = fn or self._default_fn

        if fn is None:
            passed = any(vr.passed for vr in verifier_results)
            value = 1.0 if passed else 0.0
            return RewardBreakdown(
                total_reward=value,
                components=[RewardComponent(name="task_success", value=value)],
            )

        return fn(state, trajectory, verifier_results, task)
