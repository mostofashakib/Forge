from __future__ import annotations

from collections.abc import Sequence

import pytest

from forge.contracts import Observation, PromptTemplate, Task, ToolSpec


class _Minimal(PromptTemplate):
    def system(self, task: Task) -> str:
        return f"Goal: {task.objective}"

    def user(self, observation: Observation, task: Task) -> str:
        return observation.text or str(observation.payload)

    def tool_descriptions(self, tools: Sequence[ToolSpec]) -> list[dict]:
        return [{"name": t.name, "description": t.description} for t in tools]


def test_a_template_renders_system_user_and_tools():
    template = _Minimal()
    task = Task(id="t1", objective="close the ticket")
    assert template.system(task) == "Goal: close the ticket"
    assert template.user(Observation(text="state"), task) == "state"
    assert template.tool_descriptions([ToolSpec(name="close")]) == [
        {"name": "close", "description": ""}
    ]


def test_no_tools_renders_no_descriptions():
    # False-positive guard: an env with no tools must not get a placeholder tool.
    assert _Minimal().tool_descriptions([]) == []


def test_a_template_missing_a_method_cannot_be_instantiated():
    class Incomplete(PromptTemplate):
        def system(self, task: Task) -> str:
            return ""

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
