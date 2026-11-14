"""The rows Forge exports for training, and the text format of the steps inside them.

Episode exports and calibration exports both build their rows from these
types and render steps with these functions, so the two cannot drift apart.
An export that needs more fields subclasses a row and adds them.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel

from forge.contracts.rollout import RolloutRecord

_SHELL_PROMPT = "$ "


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RolloutRow(RolloutRecord):
    """One graded attempt, a GRPO row. Rows of one task form its group."""

    agent_id: str
    total_steps: int


class PreferencePair(BaseModel):
    """A better and a worse answer to the same prompt."""

    env_name: str
    task: str
    chosen: list[ChatMessage]
    rejected: list[ChatMessage]
    chosen_reward: float
    rejected_reward: float
    chosen_passed: bool
    rejected_passed: bool


class SftStep(BaseModel):
    """One step of a successful attempt: the prompt and earlier actions, then the action taken."""

    messages: list[ChatMessage]
    episode_id: str
    step_index: int
    total_reward: float


class FailureEpisode(BaseModel):
    """A failed attempt with every step it took."""

    episode_id: str
    env_name: str
    task_name: str
    total_reward: float
    total_steps: int
    steps: list[dict]


def shell_line(command: str) -> str:
    return f"{_SHELL_PROMPT}{command}"


def step_line(step: Mapping) -> str:
    """A step as `{"tool", "args"}`: a shell command, or the tool with its arguments when it has any."""
    args = step.get("args") or {}
    if step["tool"] == "shell":
        return shell_line(args.get("command", ""))
    return f"{step['tool']} {json.dumps(args)}" if args else step["tool"]


def steps_text(steps: Sequence[Mapping]) -> str:
    return "\n".join(step_line(step) for step in steps)


def action_command(action_raw: str) -> str:
    """A recorded episode action, stored as JSON text: its command, or the action itself."""
    try:
        action = json.loads(action_raw) if action_raw else {}
    except (json.JSONDecodeError, TypeError):
        return str(action_raw)
    return action.get("command") or action.get("cmd") or json.dumps(action)


def action_line(action_raw: str) -> str:
    return shell_line(action_command(action_raw))


def actions_text(actions_raw: Sequence[str]) -> str:
    return "\n".join(action_line(raw) for raw in actions_raw)


def exchange(prompt: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def step_exchanges(prompt: str, lines: Sequence[str]) -> list[list[dict]]:
    """One exchange per step. Each shows the prompt and the actions before it, then asks for the next."""
    exchanges = []
    for index, line in enumerate(lines):
        context = prompt
        if index:
            context += "\nPrevious actions:\n" + "\n".join(lines[:index])
        exchanges.append(exchange(context, line))
    return exchanges
