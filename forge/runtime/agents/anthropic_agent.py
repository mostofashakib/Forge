from __future__ import annotations
import json
from collections.abc import Sequence
from forge.contracts import Observation, PromptTemplate, Task, ToolSpec
from forge.runtime.agents.prompts import FORGE_AGENT_PROMPT

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


class AnthropicAgent:
    def __init__(
        self,
        model: str,
        client=None,
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
            if anthropic is None:
                raise ImportError("anthropic package not installed")
            self._client = anthropic.Anthropic()

    def act(self, obs: dict, action_types: frozenset[str]) -> dict:
        tools = self._tools(action_types)
        if self._prompt_template is not None:
            system = self._prompt_template.system(self._task)
            user = self._prompt_template.user(Observation(payload=obs), self._task)
        else:
            system = f"{FORGE_AGENT_PROMPT.system}\n\nOUTPUT FORMAT: {FORGE_AGENT_PROMPT.output_contract}"
            user = FORGE_AGENT_PROMPT.observation_template.format(
                observation=json.dumps(obs)
            )
        messages = [{
            "role": "user",
            "content": user,
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

    def _tools(self, action_types: frozenset[str]) -> list[dict]:
        specs = [self._tool_specs.get(name, ToolSpec(name=name)) for name in sorted(action_types)]
        if self._prompt_template is not None:
            return self._prompt_template.tool_descriptions(specs)
        return [
            {
                "name": spec.name,
                "description": spec.description or FORGE_AGENT_PROMPT.action_description_template.format(
                    action=spec.name
                ),
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": True},
            }
            for spec in specs
        ]


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
