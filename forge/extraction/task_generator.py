from __future__ import annotations
from forge.extraction.llm_client import LLMClient
from forge.extraction.prompts import TASK_PROMPT
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
            system=TASK_PROMPT.system,
            user=TASK_PROMPT.user_template.format(
                action_names=[a.name for a in actions],
                prompt=prompt,
            ),
            schema=TASK_PROMPT.output_type,
        )
        return result.tasks
