from __future__ import annotations
from forge.extraction.llm_client import LLMClient
from forge.extraction.prompts import ExtractionPrompts
from forge.extraction.schemas import ActionDef, EntityDef, PolicyRule


class PolicyParser:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(
        self, prompt: str, entities: list[EntityDef], actions: list[ActionDef]
    ) -> list[PolicyRule]:
        result = self._client.extract(
            system=ExtractionPrompts.POLICIES.system,
            user=ExtractionPrompts.POLICIES.user_template.format(
                action_names=[a.name for a in actions],
                prompt=prompt,
            ),
            schema=ExtractionPrompts.POLICIES.output_type,
        )
        return result.policies
