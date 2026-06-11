# tests/runtime/test_interaction.py
import copy
import pytest
from forge.runtime.env_builder import EnvBuilder
from forge.runtime.interaction import (
    BrowserUse,
    BrowserUseSchema,
    ComputerUse,
    ComputerUseSchema,
    ToolUse,
    ToolUseSchema,
)
from forge.runtime.snapshot import InvalidActionError, ToolParam, ToolSpec
from forge.runtime.transition import TransitionResult


# ---------------------------------------------------------------------------
# ToolUseSchema — contract for API endpoints / functions of the environment
# ---------------------------------------------------------------------------

def reply_schema() -> ToolUseSchema:
    return ToolUseSchema(tools=[
        ToolSpec(
            name="reply_email",
            description="Reply to a thread",
            params=[
                ToolParam(name="thread_id", type="string", required=True),
                ToolParam(name="body", type="string", required=False),
            ],
        ),
        ToolSpec(name="archive_email"),
    ])


def test_tool_schema_accepts_valid_action():
    schema = reply_schema()
    assert schema.validate_action({"type": "reply_email", "thread_id": "t_1"}) is None


def test_tool_schema_rejects_unknown_tool():
    error = reply_schema().validate_action({"type": "delete_everything"})
    assert error is not None and "delete_everything" in error


def test_tool_schema_rejects_missing_required_param():
    error = reply_schema().validate_action({"type": "reply_email"})
    assert error is not None and "thread_id" in error


def test_tool_schema_rejects_action_without_type():
    assert reply_schema().validate_action({"thread_id": "t_1"}) is not None


def test_tool_schema_lists_tool_names():
    assert reply_schema().tool_names() == ["archive_email", "reply_email"]


# ---------------------------------------------------------------------------
# ComputerUseSchema — contract for VM / OS interaction
# ---------------------------------------------------------------------------

def test_computer_schema_accepts_shell_exec():
    schema = ComputerUseSchema(os="linux")
    assert schema.validate_action({"action_type": "exec", "command": "ls -la"}) is None


def test_computer_schema_rejects_exec_without_command():
    assert ComputerUseSchema().validate_action({"action_type": "exec"}) is not None
    assert ComputerUseSchema().validate_action({"action_type": "exec", "command": ""}) is not None


def test_computer_schema_rejects_unknown_primitive():
    error = ComputerUseSchema().validate_action({"action_type": "format_disk"})
    assert error is not None and "format_disk" in error


def test_computer_schema_declares_operating_system():
    assert ComputerUseSchema().os == "linux"
    assert ComputerUseSchema(os="windows").os == "windows"
    with pytest.raises(ValueError):
        ComputerUseSchema(os="templeos")


def test_computer_schema_can_restrict_primitives():
    schema = ComputerUseSchema(actions=["screenshot"])
    assert schema.validate_action({"action_type": "exec", "command": "ls"}) is not None
    assert schema.validate_action({"action_type": "screenshot"}) is None


# ---------------------------------------------------------------------------
# BrowserUseSchema — contract for browser interaction
# ---------------------------------------------------------------------------

def test_browser_schema_accepts_valid_primitives():
    schema = BrowserUseSchema(display_width=800, display_height=600)
    assert schema.validate_action({"action_type": "click", "x": 10, "y": 10}) is None
    assert schema.validate_action({"action_type": "type", "text": "hello"}) is None
    assert schema.validate_action({"action_type": "press", "key": "Return"}) is None
    assert schema.validate_action({"action_type": "navigate", "url": "http://x"}) is None
    assert schema.validate_action({"action_type": "scroll", "delta_y": 100}) is None


def test_browser_schema_rejects_unknown_primitive():
    error = BrowserUseSchema().validate_action({"action_type": "teleport"})
    assert error is not None and "teleport" in error


def test_browser_schema_rejects_out_of_bounds_click():
    schema = BrowserUseSchema(display_width=800, display_height=600)
    assert schema.validate_action({"action_type": "click", "x": 900, "y": 10}) is not None
    assert schema.validate_action({"action_type": "click", "x": 10, "y": -5}) is not None


def test_browser_schema_rejects_navigate_without_url():
    assert BrowserUseSchema().validate_action({"action_type": "navigate"}) is not None


def test_browser_schema_can_restrict_primitives():
    schema = BrowserUseSchema(actions=["click", "type"])
    assert schema.validate_action({"action_type": "navigate", "url": "http://x"}) is not None


# ---------------------------------------------------------------------------
# ToolUse / ComputerUse / BrowserUse — validated execution
# ---------------------------------------------------------------------------

def test_tool_use_executes_valid_action_through_executor():
    calls = []
    tool_use = ToolUse(schema=reply_schema(), executor=lambda a: calls.append(a) or "done")
    result = tool_use.execute({"type": "archive_email"})
    assert result == "done"
    assert calls == [{"type": "archive_email"}]


def test_tool_use_raises_on_contract_violation_without_executing():
    calls = []
    tool_use = ToolUse(schema=reply_schema(), executor=calls.append)
    with pytest.raises(InvalidActionError):
        tool_use.execute({"type": "reply_email"})  # missing thread_id
    assert calls == []


def test_computer_use_executes_valid_exec():
    calls = []
    cu = ComputerUse(schema=ComputerUseSchema(), executor=lambda a: calls.append(a) or "ok")
    assert cu.execute({"action_type": "exec", "command": "whoami"}) == "ok"
    assert calls == [{"action_type": "exec", "command": "whoami"}]


def test_computer_use_raises_on_contract_violation():
    cu = ComputerUse(schema=ComputerUseSchema(), executor=lambda a: None)
    with pytest.raises(InvalidActionError):
        cu.execute({"action_type": "exec"})


def test_browser_use_executes_valid_primitive():
    calls = []
    bu = BrowserUse(schema=BrowserUseSchema(), executor=lambda a: calls.append(a))
    bu.execute({"action_type": "click", "x": 5, "y": 5})
    assert calls == [{"action_type": "click", "x": 5, "y": 5}]


def test_browser_use_raises_on_contract_violation():
    bu = BrowserUse(schema=BrowserUseSchema(display_width=100, display_height=100),
                    executor=lambda a: None)
    with pytest.raises(InvalidActionError):
        bu.execute({"action_type": "click", "x": 500, "y": 5})


def test_capability_names():
    assert ToolUse(schema=reply_schema(), executor=lambda a: None).name == "tool_use"
    assert ComputerUse(schema=ComputerUseSchema(), executor=lambda a: None).name == "computer_use"
    assert BrowserUse(schema=BrowserUseSchema(), executor=lambda a: None).name == "browser_use"


# ---------------------------------------------------------------------------
# Environment access — any combination of the three capabilities
# ---------------------------------------------------------------------------

class CounterFactory:
    def create(self, ctx, options):
        return {"counter": {"c_0": {"id": "c_0", "value": 0}}}


def increment(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += 1
    return TransitionResult(state=new_state, events=[])


def make_builder():
    return (
        EnvBuilder("interaction_env", domain="test", max_steps=10)
        .with_initial_state(CounterFactory())
        .with_transition("increment", increment)
    )


def test_forge_env_exposes_tool_use_capability():
    env = make_builder().build(verify=False)
    assert env.capabilities() == ["tool_use"]
    assert env.tool_use is not None
    assert env.computer_use is None
    assert env.browser_use is None
    assert env.tool_use.schema.tool_names() == ["increment"]


def test_env_tool_use_executes_against_the_env():
    env = make_builder().build(verify=False)
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.tool_use.execute({"type": "increment"})
    assert obs["counter"]["c_0"]["value"] == 1


def test_env_tool_use_rejects_unknown_tool_before_stepping():
    env = make_builder().build(verify=False)
    env.reset(seed=1)
    with pytest.raises(InvalidActionError):
        env.tool_use.execute({"type": "rm_rf"})


def test_builder_can_attach_all_three_capabilities():
    shell_calls, gui_calls = [], []
    env = (
        make_builder()
        .with_computer_use(executor=shell_calls.append, schema=ComputerUseSchema(os="linux"))
        .with_browser_use(executor=gui_calls.append)
        .build(verify=False)
    )
    assert env.capabilities() == ["tool_use", "computer_use", "browser_use"]
    env.computer_use.execute({"action_type": "exec", "command": "ls"})
    env.browser_use.execute({"action_type": "click", "x": 1, "y": 1})
    assert shell_calls == [{"action_type": "exec", "command": "ls"}]
    assert gui_calls == [{"action_type": "click", "x": 1, "y": 1}]


def test_browser_runner_builds_browser_use_for_page():
    from forge.envgen.browser_runner import BrowserEpisodeRunner

    class FakePage:
        def __init__(self):
            self.clicks = []
            self.mouse = self

        def click(self, x, y):
            self.clicks.append((x, y))

    page = FakePage()
    bu = BrowserEpisodeRunner.browser_use_for(page)
    assert bu.name == "browser_use"
    bu.execute({"action_type": "click", "x": 3, "y": 4})
    assert page.clicks == [(3, 4)]
    with pytest.raises(InvalidActionError):
        bu.execute({"action_type": "teleport"})


def test_cli_runner_exposes_computer_use_contract():
    from forge.envgen.cli_runner import CliEpisodeConfig, CliEpisodeRunner

    runner = CliEpisodeRunner(CliEpisodeConfig(container_id="c_fake", objective="test"))
    executed = []
    runner._exec = lambda command: executed.append(command) or {"command": command}
    cu = runner.computer_use()
    assert cu.name == "computer_use"
    assert cu.schema.os == "linux"
    cu.execute({"action_type": "exec", "command": "ls"})
    assert executed == ["ls"]
    with pytest.raises(InvalidActionError):
        cu.execute({"action_type": "exec"})
