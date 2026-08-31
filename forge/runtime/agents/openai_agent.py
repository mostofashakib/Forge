from __future__ import annotations
import json
from collections.abc import Sequence
from forge.contracts.prompting import PromptTemplate
from forge.contracts.types import Observation, Task, ToolSpec
from forge.runtime.agents.prompts import FORGE_AGENT_PROMPT

try:
    import openai
except ImportError:
    openai = None  # type: ignore


class OpenAIAgent:
    def __init__(
        self,
        model: str,
        client=None,
        base_url: str | None = None,
        logger=None,
        prompt_template: PromptTemplate | None = None,
        task: Task | None = None,
        tool_specs: Sequence[ToolSpec] | None = None,
    ) -> None:
        self._model = model
        self.logger = logger
        self._prompt_template = prompt_template
        self._task = task or Task(id="runtime", objective="advance the workflow")
        self._tool_specs = {spec.name: spec for spec in (tool_specs or ())}
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
        tools = self._tools(action_types)
        messages = self._messages(obs)
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

    def _tools(self, action_types: frozenset[str]) -> list[dict]:
        if self._prompt_template is not None:
            descriptions = self._prompt_template.tool_descriptions(
                [self._tool_specs.get(name, ToolSpec(name=name)) for name in sorted(action_types)]
            )
            return [
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item["description"],
                        "parameters": item["input_schema"],
                    },
                }
                for item in descriptions
            ]
        return [
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

    def _messages(self, obs: dict) -> list[dict]:
        if self._prompt_template is not None:
            return [
                {"role": "system", "content": self._prompt_template.system(self._task)},
                {
                    "role": "user",
                    "content": self._prompt_template.user(
                        Observation(payload=obs), self._task
                    ),
                },
            ]
        return [
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


def _serialize_choice(choice) -> dict:
    """Render an OpenAI chat choice into a JSON-serializable dict."""
    message = choice.message
    tool_calls = [
        {"name": tc.function.name, "arguments": tc.function.arguments}
        for tc in (message.tool_calls or [])
    ]
    return {"content": getattr(message, "content", None), "tool_calls": tool_calls}
