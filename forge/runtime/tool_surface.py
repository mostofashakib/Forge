"""What an in-process environment lets the agent do, grouped by interaction mode."""
from __future__ import annotations

from collections.abc import Iterable

from forge.runtime.control import SUBMIT_ACTION
from forge.runtime.snapshot import ToolSpec


class ToolSurface:
    """The agent's action space: core tool calls plus any attached modalities.

    `modalities` are the optional MCP / REST / oRPC / OS / browser attachments,
    each exposing `name` and `schema.tool_specs()`. Absent ones are None.
    """

    def __init__(self, action_types: Iterable[str], tool_specs: dict[str, ToolSpec], modalities: Iterable) -> None:
        self._action_types = frozenset(action_types)
        self._tool_specs = tool_specs
        self._modalities = [modality for modality in modalities if modality is not None]

    def tool_specs(self) -> list[ToolSpec]:
        """Every tool the agent may call, with params — bare spec if undocumented."""
        specs = []
        for name in sorted(self._action_types):
            if name == SUBMIT_ACTION:
                specs.append(ToolSpec(
                    name=SUBMIT_ACTION,
                    description="Finish the episode and grade the current state",
                ))
            else:
                specs.append(self._tool_specs.get(name, ToolSpec(name=name)))
        return specs

    def capabilities(self) -> list[str]:
        """Interaction modes available: always `tool_use`, then each attached modality."""
        return ["tool_use", *(modality.name for modality in self._modalities)]

    def by_modality(self) -> dict[str, list[ToolSpec]]:
        """Every action the agent can take, grouped by interaction modality."""
        surface: dict[str, list[ToolSpec]] = {"tool_use": self.tool_specs()}
        for modality in self._modalities:
            surface[modality.name] = modality.schema.tool_specs()
        return surface
