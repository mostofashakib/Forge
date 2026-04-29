from __future__ import annotations
from pydantic import BaseModel
from forge.extraction.llm_client import LLMClient
from forge.extraction.schemas import EntityDef

_SYSTEM = """Extract all data entities from the description.
For each entity provide: name (singular snake_case), primary_key (usually "id"),
and fields with types (string, integer, boolean, enum, list).
For enum fields include all possible values."""


class _EntityExtractionResult(BaseModel):
    entities: list[EntityDef]


class EntityExtractor:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, prompt: str) -> list[EntityDef]:
        result = self._client.extract(
            system=_SYSTEM,
            user=f"Extract entities from:\n\n{prompt}",
            schema=_EntityExtractionResult,
        )
        return result.entities
