"""Concrete task sources for in-process environments."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from forge.contracts import Task, TaskSource


class StaticTaskSource(TaskSource):
    """An immutable, ordered collection of tasks declared at build time."""

    def __init__(self, tasks: Iterable[Task | dict] = ()) -> None:
        normalized = tuple(self._normalize(task, index) for index, task in enumerate(tasks))
        self._tasks = normalized
        self._by_id = {task.id: task for task in normalized}
        if len(self._by_id) != len(normalized):
            raise ValueError("task ids must be unique")

    def tasks(self) -> Sequence[Task]:
        return self._tasks

    def get(self, task_id: str) -> Task:
        try:
            return self._by_id[task_id]
        except KeyError:
            raise KeyError(task_id) from None

    @staticmethod
    def _normalize(task: Task | dict, index: int) -> Task:
        if isinstance(task, Task):
            return task
        raw = dict(task)
        task_id = str(raw.pop("id", raw.pop("name", f"task-{index}")))
        objective = str(raw.pop("objective", raw.pop("description", task_id)))
        known = {
            key: raw.pop(key)
            for key in ("seed", "success_conditions", "failure_conditions")
            if key in raw
        }
        return Task(id=task_id, objective=objective, metadata=raw, **known)
