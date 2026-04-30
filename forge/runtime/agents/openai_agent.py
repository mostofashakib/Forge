from __future__ import annotations
import json
from forge.runtime.agents.prompts import FORGE_AGENT_PROMPT

try:
    import openai
except ImportError:
    openai = None  # type: ignore


class OpenAIAgent:
    def __init__(self, model: str, client=None, base_url: str | None = None) -> None:
        self._model = model
        if client is not None:
            self._client = client
        else:
            if openai is None:
                raise ImportError("openai package not installed")
            kwargs = {}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)

    def act(self, obs: dict, action_types: frozenset[str]) -> dict:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": at,
                    "description": FORGE_AGENT_PROMPT.action_description_template.format(action=at),
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
                },
            }
            for at in sorted(action_types)
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            tools=tools,
            messages=[
                {"role": "system", "content": FORGE_AGENT_PROMPT.system},
                {
                    "role": "user",
                    "content": FORGE_AGENT_PROMPT.observation_template.format(
                        observation=json.dumps(obs)
                    ),
                },
            ],
        )
        choice = response.choices[0]
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            return {"type": tc.function.name, **json.loads(tc.function.arguments)}
        return {"type": sorted(action_types)[0]}
