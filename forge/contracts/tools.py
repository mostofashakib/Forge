"""What the model can do in the world."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from forge.contracts.types import ToolSpec


class ToolProvider(ABC):
    """The set of tools an environment exposes to the agent.

    Optional on the Environment facade: a shell environment exposes a command
    line rather than a tool schema.
    """

    @abstractmethod
    def tools(self) -> Sequence[ToolSpec]:
        """Every tool the agent may call, in a stable order."""
