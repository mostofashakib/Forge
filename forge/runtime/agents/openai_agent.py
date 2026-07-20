from __future__ import annotations
import json
from forge.runtime.agents.prompts import FORGE_AGENT_PROMPT

try:
    import openai
except ImportError:
    openai = None  # type: ignore


class OpenAIAgent:
    def __init__(self, model: str, client=None, base_url: str | None = None, logger=None) -> None:
        self._model = model
        self.logger = logger
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
        messages = [
            {
                "role": "system",
                "content": (
                    f"{FORGE_AGENT_PROMPT.system}\n\n"
                    f"OUTPUT FORMAT: {FORGE_AGENT_PROMPT.output_contract}"
                ),
            },
            {
                "role": "user",
                "content": FORGE_AGENT_PROMPT.observation_template.format(
                    observation=json.dumps(obs)
                ),
            },
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            tools=tools,
            messages=messages,
        )
        choice = response.choices[0]
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            action = {"type": tc.function.name, **json.loads(tc.function.arguments)}
        else:
            action = {"type": sorted(action_types)[0]}
        if self.logger is not None:
            self.logger.log_llm_call(
                prompt={"messages": messages,
                        "tools": [t["function"]["name"] for t in tools]},
                tool_call=action,
                response=_serialize_choice(choice),
            )
        return action


def _serialize_choice(choice) -> dict:
    """Render an OpenAI chat choice into a JSON-serializable dict."""
    message = choice.message
    tool_calls = [
        {"name": tc.function.name, "arguments": tc.function.arguments}
        for tc in (message.tool_calls or [])
    ]
    return {"content": getattr(message, "content", None), "tool_calls": tool_calls}
