"""Named prompt interface for tracker rollout roles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptDefinition:
    prompt_id: str
    description: str
    system_template: str


PROMPTS: tuple[PromptDefinition, ...] = (
    PromptDefinition(
        prompt_id="tracker_operator",
        description="General-purpose, least-privilege task-tracker operator.",
        system_template=(
            "You are operating a deterministic task tracker as the user assigned to "
            "the episode. Inspect the workspace before making changes. Treat tool "
            "results as the source of truth, preserve state across turns, and make "
            "only changes authorized by the user instruction. Read a record before "
            "you replace any list-valued field on it, and verify final state before "
            "answering."
        ),
    ),
    PromptDefinition(
        prompt_id="handover_coordinator",
        description="Reassignment and workload-handover coordinator.",
        system_template=(
            "You coordinate work handovers in this tracker. Enumerate the complete "
            "set of affected records before mutating any of them, decide each one on "
            "its own fields rather than on the group's, and leave every unrelated "
            "record exactly as you found it. Report what you changed when you are "
            "done."
        ),
    ),
    PromptDefinition(
        prompt_id="delivery_manager",
        description="Milestone and dependency-aware delivery manager.",
        system_template=(
            "You manage delivery for this tracker. Inspect projects, milestones and "
            "dependency edges before changing task state, respect the status machine, "
            "and never close work that something else still waits on. Confirm the "
            "resulting plan before answering."
        ),
    ),
)


class PromptInterface:
    """Registry and renderer for named, reusable rollout prompts."""

    def __init__(self, definitions: tuple[PromptDefinition, ...] = PROMPTS) -> None:
        self._definitions = {item.prompt_id: item for item in definitions}
        if not self._definitions:
            raise ValueError("At least one prompt definition is required.")

    def list(self) -> list[dict[str, str]]:
        return [
            {"prompt_id": item.prompt_id, "description": item.description}
            for item in self._definitions.values()
        ]

    def render(
        self,
        prompt_id: str,
        *,
        user_instruction: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        definition = self._definitions.get(prompt_id)
        if definition is None:
            available = ", ".join(self._definitions)
            raise KeyError(f"Unknown prompt_id {prompt_id!r}. Available prompts: {available}")
        system = definition.system_template
        if context:
            lines = "\n".join(f"- {key}: {value}" for key, value in sorted(context.items()))
            system = f"{system}\n\nEpisode context:\n{lines}"
        return {
            "prompt_id": prompt_id,
            "description": definition.description,
            "system": system,
            "user": user_instruction,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_instruction},
            ],
        }


def get_prompt_interface() -> PromptInterface:
    return PromptInterface()
