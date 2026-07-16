from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar
from pydantic import BaseModel
from forge.extraction.schemas import EntityDef, ActionDef, PolicyRule, TaskTemplate

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ExtractionPrompt(Generic[T]):
    """
    Bundles a system prompt, a user message template, and the Pydantic model
    the LLM must populate.  Having output_type here makes the expected schema
    visible at the prompt-definition site rather than scattered across callers.
    """
    system: str
    user_template: str  # keyword placeholders filled by each extractor
    output_type: type[T]


# ── Output wrappers ────────────────────────────────────────────────────────────

class EntityExtractionResult(BaseModel):
    entities: list[EntityDef]


class ActionExtractionResult(BaseModel):
    actions: list[ActionDef]


class PolicyExtractionResult(BaseModel):
    policies: list[PolicyRule]


class TaskExtractionResult(BaseModel):
    tasks: list[TaskTemplate]


# ── Prompt objects ─────────────────────────────────────────────────────────────

ENTITY_PROMPT: ExtractionPrompt[EntityExtractionResult] = ExtractionPrompt(
    system=(
        "Extract all data entities from the workflow description.\n"
        "For each entity provide:\n"
        "  - name: singular snake_case identifier\n"
        "  - primary_key: usually \"id\"\n"
        "  - fields: list of {name, type, values (enum only), nullable, default}\n"
        "Supported field types: string, integer, boolean, enum, list, dict.\n"
        "Call the extract tool with all results. Every field must be populated."
    ),
    user_template="Extract entities from:\n\n{prompt}",
    output_type=EntityExtractionResult,
)

ACTION_PROMPT: ExtractionPrompt[ActionExtractionResult] = ExtractionPrompt(
    system=(
        "Infer all actions an agent can perform in this workflow system.\n"
        "For each action provide:\n"
        "  - name: snake_case verb_noun (e.g. assign_ticket, close_order)\n"
        "  - params: list of {name, type, values (enum only), nullable, default}\n"
        "  - mutates: entity fields changed (format: entity_name.field_name)\n"
        "  - requires_permission: roles that must be present\n"
        "Call the extract tool with all results. Every field must be populated."
    ),
    user_template="Entities:\n{entity_summary}\n\nDescription:\n{prompt}",
    output_type=ActionExtractionResult,
)

POLICY_PROMPT: ExtractionPrompt[PolicyExtractionResult] = ExtractionPrompt(
    system=(
        "Extract workflow policies and business-rule constraints.\n"
        "For each policy provide:\n"
        "  - id: snake_case identifier\n"
        "  - condition: Python boolean expression evaluated against state snapshot\n"
        "  - forbidden_actions: action names that violate this policy\n"
        "  - description: plain-English explanation\n"
        "Call the extract tool with all results (use an empty list if no policies exist)."
    ),
    user_template="Actions: {action_names}\n\nDescription:\n{prompt}",
    output_type=PolicyExtractionResult,
)

TASK_PROMPT: ExtractionPrompt[TaskExtractionResult] = ExtractionPrompt(
    system=(
        "Generate RL task templates for this workflow environment.\n"
        "For each task provide:\n"
        "  - name: snake_case task identifier\n"
        "  - description: what the agent must accomplish\n"
        "  - success_conditions: list of {type, expression, rubric, description}\n"
        "    Condition types: state_check, event_check, temporal_check, negative_check\n"
        "  - failure_conditions: list in the same format (optional)\n"
        "Call the extract tool with all results. Every field must be populated."
    ),
    user_template="Actions: {action_names}\n\nDescription:\n{prompt}",
    output_type=TaskExtractionResult,
)


class ExtractionPrompts:
    """Central prompt catalog for the compiler extraction pipeline."""

    ENTITIES = ENTITY_PROMPT
    ACTIONS = ACTION_PROMPT
    POLICIES = POLICY_PROMPT
    TASKS = TASK_PROMPT
