"""The default PromptTemplate must reproduce the existing agent prompt exactly.

`ForgeAgentPromptTemplate` wraps `FORGE_AGENT_PROMPT` rather than restating it.
Every assertion below compares against that constant instead of a literal, so a
copy-pasted implementation fails the moment the constant is edited — which is
the regression this file exists to catch.
"""
from __future__ import annotations

import json

import pytest

from forge.contracts import Observation, PromptTemplate, Task, ToolParam, ToolSpec
from forge.runtime.agents.prompts import FORGE_AGENT_PROMPT
from forge.runtime.prompting import ForgeAgentPromptTemplate


def _task() -> Task:
    return Task(id="t1", objective="Close the overdue ticket")


def test_the_default_template_is_a_prompt_template():
    assert isinstance(ForgeAgentPromptTemplate(), PromptTemplate)


def test_the_system_prompt_is_the_shared_constant_verbatim():
    assert ForgeAgentPromptTemplate().system(_task()) == FORGE_AGENT_PROMPT.system


def test_the_user_message_renders_the_observation_into_the_template():
    observation = Observation(payload={"tickets": {"t_1": {"status": "open"}}})

    rendered = ForgeAgentPromptTemplate().user(observation, _task())

    expected = FORGE_AGENT_PROMPT.observation_template.format(
        observation=json.dumps(observation.payload, sort_keys=True)
    )
    assert rendered == expected


def test_the_user_message_carries_the_observation_payload():
    # False-positive guard: the template could render and still drop the state.
    # Without this, an implementation returning the bare template would pass.
    observation = Observation(payload={"tickets": {"t_1": {"status": "overdue"}}})

    rendered = ForgeAgentPromptTemplate().user(observation, _task())

    assert "overdue" in rendered
    assert "{observation}" not in rendered


def test_tool_descriptions_yield_one_entry_per_tool():
    tools = [
        ToolSpec(name="close_ticket", params=[ToolParam(name="ticket_id")]),
        ToolSpec(name="reply", description="Reply to the customer"),
    ]

    described = ForgeAgentPromptTemplate().tool_descriptions(tools)

    assert [entry["name"] for entry in described] == ["close_ticket", "reply"]


def test_a_tool_without_a_description_falls_back_to_the_shared_template():
    tools = [ToolSpec(name="close_ticket")]

    described = ForgeAgentPromptTemplate().tool_descriptions(tools)

    assert described[0]["description"] == (
        FORGE_AGENT_PROMPT.action_description_template.format(action="close_ticket")
    )


def test_a_tools_own_description_wins_over_the_fallback():
    # False-positive guard: the fallback must not overwrite a real description.
    tools = [ToolSpec(name="reply", description="Reply to the customer")]

    described = ForgeAgentPromptTemplate().tool_descriptions(tools)

    assert described[0]["description"] == "Reply to the customer"


def test_tool_parameters_become_a_json_schema_object():
    tools = [
        ToolSpec(
            name="close_ticket",
            params=[
                ToolParam(name="ticket_id", type="string", description="Which ticket"),
                ToolParam(name="notify", type="boolean", required=False),
            ],
        )
    ]

    schema = ForgeAgentPromptTemplate().tool_descriptions(tools)[0]["input_schema"]

    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"ticket_id", "notify"}
    assert schema["properties"]["ticket_id"]["description"] == "Which ticket"
    # Only the required parameter is listed, so an optional one is not demanded.
    assert schema["required"] == ["ticket_id"]


def test_no_tools_produces_no_descriptions():
    # Negative case: an environment with no tool surface must not synthesize one.
    assert ForgeAgentPromptTemplate().tool_descriptions([]) == []


def test_an_incomplete_subclass_cannot_be_instantiated():
    # Negative case: PromptTemplate is prescriptive, not advisory.
    class Partial(ForgeAgentPromptTemplate.__bases__[0]):  # PromptTemplate
        def system(self, task):
            return ""

    with pytest.raises(TypeError):
        Partial()
