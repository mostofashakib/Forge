from __future__ import annotations
from forge.extraction.llm_client import LLMClient
from forge.extraction.prompts import ENTITY_PROMPT
from forge.extraction.schemas import EntityDef


class EntityExtractor:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, prompt: str) -> list[EntityDef]:
        result = self._client.extract(
            system=ENTITY_PROMPT.system,
            user=ENTITY_PROMPT.user_template.format(prompt=prompt),
            schema=ENTITY_PROMPT.output_type,
        )
        return result.entities
