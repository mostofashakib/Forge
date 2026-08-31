"""How the task is presented to the model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from forge.contracts.types import Observation, Task, ToolSpec


class PromptTemplate(ABC):
    """Renders the text an LLM agent sees each turn.

    Optional on the Environment facade: an environment driven by a trainer that
    supplies its own prompting has no template of its own.
    """

    @abstractmethod
    def system(self, task: Task) -> str:
        """The system prompt for this task."""

    @abstractmethod
    def user(self, observation: Observation, task: Task) -> str:
        """The per-turn user message carrying the current observation."""

    @abstractmethod
    def tool_descriptions(self, tools: Sequence[ToolSpec]) -> list[dict]:
        """Provider-agnostic tool descriptions for the given tool surface."""
