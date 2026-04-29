from __future__ import annotations
from pydantic import BaseModel
from forge.extraction.llm_client import LLMClient
from forge.extraction.schemas import ActionDef, EntityDef

_SYSTEM = """Infer all actions an agent can perform in this system.
For each action provide: name (snake_case verb_noun), params with types,
and which entity fields it mutates (format: entity_name.field_name)."""


class _ActionExtractionResult(BaseModel):
    actions: list[ActionDef]


class ActionInferencer:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(self, prompt: str, entities: list[EntityDef]) -> list[ActionDef]:
        entity_summary = "\n".join(
            f"- {e.name}: {[f.name for f in e.fields]}" for e in entities
        )
        result = self._client.extract(
            system=_SYSTEM,
            user=f"Entities:\n{entity_summary}\n\nDescription:\n{prompt}",
            schema=_ActionExtractionResult,
        )
        return result.actions
