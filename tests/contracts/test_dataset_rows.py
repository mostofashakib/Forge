"""The rows Forge exports, and the text format of the steps inside them."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge.contracts import RolloutRecord
from forge.contracts.dataset_rows import (
    FailureEpisode,
    PreferencePair,
    RolloutRow,
    SftStep,
    action_command,
    action_line,
    actions_text,
    exchange,
    shell_line,
    step_exchanges,
    step_line,
    steps_text,
)


def test_a_shell_line_is_the_command_after_a_prompt():
    assert shell_line("ls /tmp") == "$ ls /tmp"


@pytest.mark.parametrize("step, line", [
    ({"tool": "shell", "args": {"command": "ls"}}, "$ ls"),
    ({"tool": "fill", "args": {"selector": "#a", "value": "x"}}, 'fill {"selector": "#a", "value": "x"}'),
    ({"tool": "/archive", "args": {}}, "/archive"),
    ({"tool": "/archive"}, "/archive"),
])
def test_a_step_is_a_shell_line_or_its_tool_with_any_arguments(step, line):
    assert step_line(step) == line


def test_steps_text_puts_one_step_on_each_line():
    steps = [{"tool": "shell", "args": {"command": "ls"}}, {"tool": "/archive", "args": {}}]

    assert steps_text(steps) == "$ ls\n/archive"


@pytest.mark.parametrize("raw, command", [
    ('{"command": "ls"}', "ls"),
    ('{"cmd": "pwd"}', "pwd"),
    ('{"type": "click"}', '{"type": "click"}'),
    ("", "{}"),
    ("not json", "not json"),
])
def test_a_recorded_action_reads_as_its_command_or_its_json(raw, command):
    assert action_command(raw) == command
    assert action_line(raw) == f"$ {command}"


def test_actions_text_puts_one_recorded_action_on_each_line():
    assert actions_text(['{"command": "ls"}', '{"type": "click"}']) == '$ ls\n$ {"type": "click"}'


def test_an_exchange_is_the_prompt_then_the_answer():
    assert exchange("do it", "$ ls") == [
        {"role": "user", "content": "do it"}, {"role": "assistant", "content": "$ ls"},
    ]


def test_each_step_exchange_shows_the_actions_before_it():
    assert step_exchanges("Task", ["$ a", "$ b"]) == [
        exchange("Task", "$ a"),
        exchange("Task\nPrevious actions:\n$ a", "$ b"),
    ]
    assert step_exchanges("Task", []) == []


def test_a_rollout_row_is_a_rollout_record_with_its_agent_and_step_count():
    row = RolloutRow(episode_id="e1", agent_id="ollama:m", total_steps=3, steps=3)

    assert list(RolloutRow.model_fields) == [*RolloutRecord.model_fields, "agent_id", "total_steps"]
    assert (row.agent_id, row.total_steps) == ("ollama:m", 3)


@pytest.mark.parametrize("model", [RolloutRow, PreferencePair, SftStep, FailureEpisode])
def test_a_row_missing_a_required_field_is_refused(model):
    with pytest.raises(ValidationError):
        model()


def test_a_preference_pair_holds_two_exchanges_for_one_task():
    pair = PreferencePair(
        env_name="ops", task="t-1", chosen=exchange("p", "$ a"), rejected=exchange("p", "$ b"),
        chosen_reward=1.0, rejected_reward=0.0, chosen_passed=True, rejected_passed=False,
    )

    assert pair.model_dump()["chosen"][1] == {"role": "assistant", "content": "$ a"}


def test_a_chat_message_must_come_from_the_user_or_the_assistant():
    with pytest.raises(ValidationError):
        SftStep(messages=[{"role": "system", "content": "x"}], episode_id="e", step_index=0, total_reward=1.0)
