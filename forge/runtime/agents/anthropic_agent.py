from __future__ import annotations
import json
from forge.runtime.agents.prompts import FORGE_AGENT_PROMPT

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


class AnthropicAgent:
    def __init__(self, model: str, client=None) -> None:
        self._model = model
        if client is not None:
            self._client = client
        else:
            if anthropic is None:
                raise ImportError("anthropic package not installed")
            self._client = anthropic.Anthropic()

    def act(self, obs: dict, action_types: frozenset[str]) -> dict:
        tools = [
            {
                "name": at,
                "description": FORGE_AGENT_PROMPT.action_description_template.format(action=at),
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": True},
            }
            for at in sorted(action_types)
        ]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=f"{FORGE_AGENT_PROMPT.system}\n\nOUTPUT FORMAT: {FORGE_AGENT_PROMPT.output_contract}",
            tools=tools,
            messages=[{
                "role": "user",
                "content": FORGE_AGENT_PROMPT.observation_template.format(
                    observation=json.dumps(obs)
                ),
            }],
        )
        for block in response.content:
            if block.type == "tool_use":
                return {"type": block.name, **block.input}
        return {"type": sorted(action_types)[0]}
