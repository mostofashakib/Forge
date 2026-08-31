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
from dataclasses import dataclass

from forge.contracts import Observation, PromptTemplate, Task, ToolSpec


@dataclass(frozen=True)
class AgentPrompt:
    """All provider-independent text used to request one agent action."""

    system: str
    observation_template: str
    action_description_template: str
    output_contract: str


FORGE_AGENT_PROMPT = AgentPrompt(
    system=(
        "You are an agent operating inside a simulated workflow environment.\n"
        "Each turn you receive the current environment state as JSON and must choose\n"
        "exactly one action by calling the corresponding tool.\n"
        "Do not produce free-form text — always respond with a tool call."
    ),
    observation_template=(
        "Current environment state:\n{observation}\n\n"
        "Select the action that best advances the workflow goal."
    ),
    action_description_template=(
        "Execute the '{action}' step in the current workflow state."
    ),
    output_contract=(
        "Response must be a single tool call matching one of the available action tools. "
        "Tool input fields must conform to the action's parameter schema."
    ),
)


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
