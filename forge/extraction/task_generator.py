from __future__ import annotations
from pydantic import BaseModel
from forge.extraction.llm_client import LLMClient
from forge.extraction.schemas import ActionDef, EntityDef, PolicyRule, TaskTemplate

_SYSTEM = """Generate RL task templates for this environment.
Each task has: name (snake_case), description, and success_conditions.
Each success condition has type (state_check, event_check, temporal_check, negative_check)
and expression (Python-like condition string)."""


class _TaskExtractionResult(BaseModel):
    tasks: list[TaskTemplate]


class TaskGenerator:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(
        self,
        prompt: str,
        entities: list[EntityDef],
        actions: list[ActionDef],
        policies: list[PolicyRule],
    ) -> list[TaskTemplate]:
        action_names = [a.name for a in actions]
        result = self._client.extract(
            system=_SYSTEM,
            user=f"Actions: {action_names}\n\nDescription:\n{prompt}",
            schema=_TaskExtractionResult,
        )
        return result.tasks
