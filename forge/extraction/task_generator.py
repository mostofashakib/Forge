from __future__ import annotations
from forge.extraction.llm_client import LLMClient
from forge.extraction.prompts import ExtractionPrompts
from forge.extraction.schemas import ActionDef, EntityDef, PolicyRule, TaskTemplate


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
        result = self._client.extract(
            system=ExtractionPrompts.TASKS.system,
            user=ExtractionPrompts.TASKS.user_template.format(
                action_names=[a.name for a in actions],
                prompt=prompt,
            ),
            schema=ExtractionPrompts.TASKS.output_type,
        )
        return result.tasks
