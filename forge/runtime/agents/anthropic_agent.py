from __future__ import annotations
import json
from forge.runtime.agents.prompts import FORGE_AGENT_PROMPT

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


class AnthropicAgent:
    def __init__(self, model: str, client=None, logger=None) -> None:
        self._model = model
        self.logger = logger
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
        system = f"{FORGE_AGENT_PROMPT.system}\n\nOUTPUT FORMAT: {FORGE_AGENT_PROMPT.output_contract}"
        messages = [{
            "role": "user",
            "content": FORGE_AGENT_PROMPT.observation_template.format(
                observation=json.dumps(obs)
            ),
        }]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )
        action = None
        for block in response.content:
            if block.type == "tool_use":
                action = {"type": block.name, **block.input}
                break
        if action is None:
            action = {"type": sorted(action_types)[0]}
        if self.logger is not None:
            self.logger.log_llm_call(
                prompt={"system": system, "messages": messages,
                        "tools": [t["name"] for t in tools]},
                tool_call=action,
                response=_serialize_content(response),
            )
        return action


def _serialize_content(response) -> list[dict]:
    """Render Anthropic response content blocks into JSON-serializable dicts."""
    blocks = []
    for block in getattr(response, "content", []) or []:
        kind = getattr(block, "type", None)
        if kind == "tool_use":
            blocks.append({"type": "tool_use", "name": block.name, "input": block.input})
        elif kind == "text":
            blocks.append({"type": "text", "text": block.text})
        else:
            blocks.append({"type": str(kind)})
    return blocks
