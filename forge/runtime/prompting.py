# forge/runtime/prompting.py
"""The default `PromptTemplate` for LLM agents.

`AgentPrompt` and its `FORGE_AGENT_PROMPT` instance already held every string
an agent sees; what they lacked was a way to hand that prompting to something
that only knows the contract. This adapts one to the other. The strings stay in
`agents/prompts.py` and are read from there rather than restated, so there is
exactly one place to edit them.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from forge.contracts import Observation, PromptTemplate, Task, ToolSpec
from forge.runtime.agents.prompts import FORGE_AGENT_PROMPT, AgentPrompt


class ForgeAgentPromptTemplate(PromptTemplate):
    """Renders the standard Forge agent prompt.

    Takes the `AgentPrompt` to render, defaulting to the shared
    `FORGE_AGENT_PROMPT`, so an environment can supply different wording
    without reimplementing the contract.
    """

    def __init__(self, prompt: AgentPrompt | None = None) -> None:
        self._prompt = prompt or FORGE_AGENT_PROMPT

    def system(self, task: Task) -> str:
        return self._prompt.system

    def user(self, observation: Observation, task: Task) -> str:
        # sort_keys so the same state always renders the same prompt: an
        # environment is only reproducible if what the model reads is too.
        payload = json.dumps(observation.payload, sort_keys=True)
        return self._prompt.observation_template.format(observation=payload)

    def tool_descriptions(self, tools: Sequence[ToolSpec]) -> list[dict]:
        return [self._describe(tool) for tool in tools]

    def _describe(self, tool: ToolSpec) -> dict:
        # A tool that carries its own description keeps it; the shared template
        # is a fallback for the generated tools that have none, not an override.
        description = tool.description or self._prompt.action_description_template.format(
            action=tool.name
        )
        return {
            "name": tool.name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                    }
                    for param in tool.params
                },
                "required": [param.name for param in tool.params if param.required],
            },
        }
