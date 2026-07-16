from __future__ import annotations
from forge.extraction.llm_client import LLMClient
from forge.extraction.prompts import ExtractionPrompts
from forge.extraction.schemas import ActionDef, EntityDef


class ActionInferencer:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, prompt: str, entities: list[EntityDef]) -> list[ActionDef]:
        entity_summary = "\n".join(
            f"- {e.name}: {[f.name for f in e.fields]}" for e in entities
        )
        result = self._client.extract(
            system=ExtractionPrompts.ACTIONS.system,
            user=ExtractionPrompts.ACTIONS.user_template.format(
                entity_summary=entity_summary,
                prompt=prompt,
            ),
            schema=ExtractionPrompts.ACTIONS.output_type,
        )
        return result.actions
