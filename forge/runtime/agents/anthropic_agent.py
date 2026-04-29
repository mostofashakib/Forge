from __future__ import annotations
import json
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
                "description": f"Perform action: {at}",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": True},
            }
            for at in sorted(action_types)
        ]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            tools=tools,
            messages=[{"role": "user", "content": json.dumps(obs)}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return {"type": block.name, **block.input}
        return {"type": sorted(action_types)[0]}
