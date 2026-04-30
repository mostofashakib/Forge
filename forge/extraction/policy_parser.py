from __future__ import annotations
from forge.extraction.llm_client import LLMClient
from forge.extraction.prompts import POLICY_PROMPT
from forge.extraction.schemas import ActionDef, EntityDef, PolicyRule


class PolicyParser:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract(
        self, prompt: str, entities: list[EntityDef], actions: list[ActionDef]
    ) -> list[PolicyRule]:
        result = self._client.extract(
            system=POLICY_PROMPT.system,
            user=POLICY_PROMPT.user_template.format(
                action_names=[a.name for a in actions],
                prompt=prompt,
            ),
            schema=POLICY_PROMPT.output_type,
        )
        return result.policies
