from __future__ import annotations

from collections.abc import Sequence

import pytest

from forge.contracts import Task, TaskSource


class _Fixed(TaskSource):
    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks

    def tasks(self) -> Sequence[Task]:
        return list(self._tasks)

    def get(self, task_id: str) -> Task:
        for task in self._tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)


def test_a_task_source_lists_and_looks_up_tasks():
    source = _Fixed([Task(id="t1", objective="close the ticket")])
    assert [t.id for t in source.tasks()] == ["t1"]
    assert source.get("t1").objective == "close the ticket"


def test_an_unknown_task_id_raises():
    # Negative: a miss must raise, not return a default task.
    source = _Fixed([Task(id="t1", objective="close the ticket")])
    with pytest.raises(KeyError):
        source.get("nope")


def test_an_empty_task_source_stays_empty():
    # False-positive guard: no tasks means no tasks, not a synthesized one.
    assert list(_Fixed([]).tasks()) == []


def test_a_task_source_missing_get_cannot_be_instantiated():
    class Incomplete(TaskSource):
        def tasks(self) -> Sequence[Task]:
            return []

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
