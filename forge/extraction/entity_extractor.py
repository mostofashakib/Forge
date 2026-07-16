from __future__ import annotations
from forge.extraction.llm_client import LLMClient
from forge.extraction.prompts import ExtractionPrompts
from forge.extraction.schemas import EntityDef


class EntityExtractor:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, prompt: str) -> list[EntityDef]:
        result = self._client.extract(
            system=ExtractionPrompts.ENTITIES.system,
            user=ExtractionPrompts.ENTITIES.user_template.format(prompt=prompt),
            schema=ExtractionPrompts.ENTITIES.output_type,
        )
        return result.entities
