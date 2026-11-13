"""The calibration policy: a native tool-calling agent choosing one step at a time."""
from __future__ import annotations

import pytest

from forge.runtime.errors import AgentError
from forge.taskfactory.policy import (
    OUTPUT_LIMIT,
    SUBMIT,
    AgentPolicy,
    PolicyError,
    Submit,
    Turn,
    action_tools,
    agent_policy,
)
from forge.taskfactory.profile import EnvironmentProfile, ToolInfo
from forge.taskfactory.runner import StepResult
from forge.taskfactory.schemas import (
    Check,
    GoldenStep,
    ReviewVerdict,
    SyntheticTask,
    TaskSeed,
)

STATE = EnvironmentProfile(
    env_name="mail", env_type="premade:gmail", family="state",
    tools=(
        ToolInfo(name="/archive_email", description="Archive one email", params=("email_id: string",)),
        ToolInfo(name="/label", params=("color: red|blue",)),
    ),
)
CLI = EnvironmentProfile(env_name="shell", env_type="cli", family="cli")
BROWSER = EnvironmentProfile(env_name="web", env_type="browser", family="browser")


class _Agent:
    def __init__(self, *actions) -> None:
        self.actions = list(actions)
        self.seen: list[tuple[dict, frozenset]] = []

    def act(self, obs, action_types):
        self.seen.append((obs, action_types))
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _names(profile) -> set[str]:
    return {spec.name for spec in action_tools(profile)}


def test_a_state_environment_offers_its_own_tools_with_typed_params_and_submit():
    specs = {spec.name: spec for spec in action_tools(STATE)}

    assert set(specs) == {"/archive_email", "/label", SUBMIT}
    assert specs["/archive_email"].description == "Archive one email"
    assert [(p.name, p.type) for p in specs["/archive_email"].params] == [("email_id", "string")]
    assert specs["/label"].params[0].type == "string"
    assert "red, blue" in specs["/label"].params[0].description


def test_a_cli_environment_offers_one_shell_tool_that_takes_a_command():
    specs = {spec.name: spec for spec in action_tools(CLI)}

    assert set(specs) == {"shell", SUBMIT}
    assert [p.name for p in specs["shell"].params] == ["command"]


def test_a_browser_environment_offers_the_browser_actions():
    assert _names(BROWSER) == {"goto", "click", "fill", "select", "check", "press", SUBMIT}


def test_a_tool_call_becomes_a_step_and_submit_ends_the_attempt():
    agent = _Agent({"type": "/archive_email", "email_id": "e1"}, {"type": SUBMIT})
    policy = AgentPolicy(agent, action_tools(STATE))

    first = policy.next_step([])
    second = policy.next_step([Turn(first, StepResult(ok=True, output="archived"))])

    assert first == GoldenStep(tool="/archive_email", args={"email_id": "e1"})
    assert isinstance(second, Submit)
    assert agent.seen[0][1] == frozenset({"/archive_email", "/label", SUBMIT})


def test_the_agent_sees_every_past_step_with_its_output_or_error():
    agent = _Agent({"type": SUBMIT})
    history = [
        Turn(GoldenStep(tool="shell", args={"command": "ls"}), StepResult(ok=True, output="a.txt\n")),
        Turn(GoldenStep(tool="shell", args={"command": "cat b"}), StepResult(ok=False, error="exited 1: no b")),
    ]

    AgentPolicy(agent, action_tools(CLI)).next_step(history)

    seen = agent.seen[0][0]["history"]
    assert seen[0] == {"tool": "shell", "args": {"command": "ls"}, "ok": True, "output": "a.txt\n"}
    assert seen[1]["ok"] is False
    assert seen[1]["error"] == "exited 1: no b"


def test_long_output_is_cut_to_the_limit_and_says_so():
    agent = _Agent({"type": SUBMIT})
    big = "x" * (OUTPUT_LIMIT + 500)

    AgentPolicy(agent, action_tools(CLI)).next_step(
        [Turn(GoldenStep(tool="shell", args={"command": "cat big"}), StepResult(ok=True, output=big))]
    )

    output = agent.seen[0][0]["history"][0]["output"]
    assert output.startswith("x" * OUTPUT_LIMIT)
    assert "500 more characters cut" in output
    assert len(output) < OUTPUT_LIMIT + 100


def test_a_tool_the_environment_lacks_is_rejected():
    policy = AgentPolicy(_Agent({"type": "/delete_everything"}), action_tools(STATE))

    with pytest.raises(PolicyError, match="/delete_everything"):
        policy.next_step([])


def test_an_agent_failure_reaches_the_caller():
    policy = AgentPolicy(_Agent(AgentError("m answered with no tool call")), action_tools(STATE))

    with pytest.raises(AgentError, match="no tool call"):
        policy.next_step([])


def test_the_agent_is_built_for_the_task_with_its_objective_and_native_tool_schemas(monkeypatch):
    import forge.taskfactory.policy as policy_module

    built: dict = {}

    def fake_make_agent(agent_id, **kwargs):
        built.update(agent_id=agent_id, **kwargs)
        return _Agent()

    monkeypatch.setattr(policy_module, "make_agent", fake_make_agent)
    task = SyntheticTask(
        id="t-00001", category="c", difficulty=1, title="Archive", objective="Archive email e1.",
        seed=TaskSeed(), golden=[GoldenStep(tool="/archive_email")], checks=[Check(kind="value", path="x", op="==", value=1)],
        reflection_points=[], step_budget=2, round=1, fingerprint="fp",
        review=ReviewVerdict(slot=0, realistic=True, realistic_reason="r", fair=True, fair_reason="f",
                             sensible=True, sensible_reason="s"),
    )

    agent_policy("ollama:qwen3.8:27b", task, STATE)

    assert built["agent_id"] == "ollama:qwen3.8:27b"
    assert built["task"].objective == "Archive email e1."
    assert {spec.name for spec in built["tool_specs"]} == _names(STATE)
    tools = {t["name"]: t for t in built["prompt_template"].tool_descriptions(built["tool_specs"])}
    assert tools["/archive_email"]["input_schema"]["properties"]["email_id"]["type"] == "string"
