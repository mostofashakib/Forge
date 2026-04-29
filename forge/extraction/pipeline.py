from __future__ import annotations
from forge.extraction.llm_client import LLMClient
from forge.extraction.schemas import CompilerInput
from forge.extraction.entity_extractor import EntityExtractor
from forge.extraction.action_inferencer import ActionInferencer
from forge.extraction.policy_parser import PolicyParser
from forge.extraction.task_generator import TaskGenerator


class ExtractionPipeline:
    def __init__(self, client: LLMClient) -> None:
        self._entity_extractor = EntityExtractor(client)
        self._action_inferencer = ActionInferencer(client)
        self._policy_parser = PolicyParser(client)
        self._task_generator = TaskGenerator(client)

    def run(self, prompt: str, project_name: str, domain: str) -> CompilerInput:
        entities = self._entity_extractor.extract(prompt)
        actions = self._action_inferencer.extract(prompt, entities)
        policies = self._policy_parser.extract(prompt, entities, actions)
        tasks = self._task_generator.extract(prompt, entities, actions, policies)
        return CompilerInput(
            project_name=project_name,
            domain=domain,
            entities=entities,
            actions=actions,
            policies=policies,
            tasks=tasks,
        )
