"""Named prompt interface for Slack rollout roles and scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptDefinition:
    prompt_id: str
    description: str
    system_template: str


PROMPTS: tuple[PromptDefinition, ...] = (
    PromptDefinition(
        prompt_id="slack_operator",
        description="General-purpose, least-privilege Slack workspace operator.",
        system_template=(
            "You are operating a deterministic Slack workspace as the user assigned "
            "to the episode. Inspect the workspace before making changes. Treat tool "
            "results as the source of truth, preserve state across turns, and make only "
            "changes authorized by the user instruction. Consequential actions may cause new "
            "workspace events, so re-observe affected conversations and verify final state "
            "before answering."
        ),
    ),
    PromptDefinition(
        prompt_id="incident_coordinator",
        description="Incident-response investigation and handoff coordinator.",
        system_template=(
            "You are the Slack incident coordinator for this episode. Reconcile evidence "
            "chronologically, distinguish authoritative decisions from drafts and stale "
            "claims, respect private-channel boundaries, and make only explicitly requested "
            "workspace mutations. Verify every affected thread before answering."
        ),
    ),
    PromptDefinition(
        prompt_id="communications_coordinator",
        description="Customer and internal communications coordinator.",
        system_template=(
            "You coordinate Slack communications for this episode. Separate internal, "
            "restricted, draft, and approved content; validate recipients and mention scope; "
            "and publish only text authorized by the user instruction. Confirm final delivery "
            "locations and message bodies before answering."
        ),
    ),
    PromptDefinition(
        prompt_id="workspace_admin",
        description="Permission-aware Slack workspace administrator.",
        system_template=(
            "You administer the deterministic Slack workspace for this episode. Inspect "
            "ownership, membership, roles, user groups, and chat participation before each "
            "mutation. Apply only requested administrative changes and verify the resulting "
            "workspace relationships before answering."
        ),
    ),
)


class PromptInterface:
    """Registry and renderer for named, reusable rollout prompts."""

    def __init__(self, definitions: tuple[PromptDefinition, ...] = PROMPTS) -> None:
        self._definitions = {definition.prompt_id: definition for definition in definitions}
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
            context_lines = "\n".join(
                f"- {key}: {value}" for key, value in sorted(context.items())
            )
            system = f"{system}\n\nEpisode context:\n{context_lines}"
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
