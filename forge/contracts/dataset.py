"""What problems the model should solve."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from forge.contracts.types import Task


class TaskSource(ABC):
    """The set of tasks an environment can pose.

    Backed by compiler TaskTemplates, an envgen ScenarioSuite, or a single
    natural-language objective, depending on the family.
    """

    @abstractmethod
    def tasks(self) -> Sequence[Task]:
        """Every task this source can pose, in a stable order."""

    @abstractmethod
    def get(self, task_id: str) -> Task:
        """One task by id. Raises KeyError when the id is unknown."""


def normalize_task(
    value: Task | Mapping[str, object], *, fallback_id: str = "task"
) -> Task:
    """Convert public task input to the canonical contract type."""
    if isinstance(value, Task):
        return value
    raw = dict(value)
    task_id = str(raw.pop("id", raw.pop("name", fallback_id)))
    objective = str(raw.pop("objective", raw.pop("description", task_id)))
    known = {
        key: raw.pop(key)
        for key in ("seed", "success_conditions", "failure_conditions")
        if key in raw
    }
    return Task(id=task_id, objective=objective, metadata=raw, **known)
