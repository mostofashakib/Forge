from __future__ import annotations
from pydantic import BaseModel
from forge.extraction.llm_client import LLMClient
from forge.extraction.schemas import ActionDef, EntityDef, PolicyRule

_SYSTEM = """Extract workflow policies and constraints.
For each policy provide: id (snake_case), condition (Python expression),
and forbidden_actions (list of action names that violate the policy)."""


class _PolicyExtractionResult(BaseModel):
    policies: list[PolicyRule]


class PolicyParser:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(
        self, prompt: str, entities: list[EntityDef], actions: list[ActionDef]
    ) -> list[PolicyRule]:
        action_names = [a.name for a in actions]
        result = self._client.extract(
            system=_SYSTEM,
            user=f"Actions: {action_names}\n\nDescription:\n{prompt}",
            schema=_PolicyExtractionResult,
        )
        return result.policies
