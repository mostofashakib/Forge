"""What problems the model should solve."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

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
