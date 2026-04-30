from __future__ import annotations
from forge.extraction.llm_client import LLMClient
from forge.extraction.prompts import ACTION_PROMPT
from forge.extraction.schemas import ActionDef, EntityDef


class ActionInferencer:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, prompt: str, entities: list[EntityDef]) -> list[ActionDef]:
        entity_summary = "\n".join(
            f"- {e.name}: {[f.name for f in e.fields]}" for e in entities
        )
        result = self._client.extract(
            system=ACTION_PROMPT.system,
            user=ACTION_PROMPT.user_template.format(
                entity_summary=entity_summary,
                prompt=prompt,
            ),
            schema=ACTION_PROMPT.output_type,
        )
        return result.actions
