"""The model under calibration, choosing one step at a time.

A `Policy` sees the steps taken so far, each with what the environment showed,
and returns the next step or `Submit`. `AgentPolicy` drives a native
tool-calling agent from `forge.runtime.agents`, so calibration uses the same
provider code as agent runs and the trajectories match how agents call tools.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from forge.contracts import AgentAdapter, Task, ToolParam, ToolSpec
from forge.runtime.agents.factory import make_agent
from forge.runtime.prompting import ForgeAgentPromptTemplate
from forge.taskfactory.profile import EnvironmentProfile
from forge.taskfactory.runner import StepResult
from forge.taskfactory.schemas import GoldenStep, SyntheticTask

SUBMIT = "submit"
# Characters of each step's output the model sees. CLI output and page HTML
# can be large, and every past step is resent each turn.
OUTPUT_LIMIT = 4000

_JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object"}
_SUBMIT_TOOL = ToolSpec(
    name=SUBMIT,
    description="Call this when the task is done. The environment's final state is graded.",
)
_CLI_TOOLS = (
    ToolSpec(
        name="shell",
        description="Run one bash command in the environment and see its output.",
        params=[ToolParam(name="command", description="The bash command to run")],
    ),
)
_SELECTOR = ToolParam(name="selector", description="CSS selector of the element")
_BROWSER_TOOLS = (
    ToolSpec(name="goto", description="Open a page on the task site.",
             params=[ToolParam(name="path", description="Path on the task site, such as /form")]),
    ToolSpec(name="click", description="Click an element.", params=[_SELECTOR]),
    ToolSpec(name="fill", description="Type a value into an input.",
             params=[_SELECTOR, ToolParam(name="value", description="Text to enter")]),
    ToolSpec(name="select", description="Choose an option in a select element.",
             params=[_SELECTOR, ToolParam(name="value", description="Option value")]),
    ToolSpec(name="check", description="Tick a checkbox.", params=[_SELECTOR]),
    ToolSpec(name="press", description="Press a key, on an element or the page.",
             params=[ToolParam(name="key", description="Key name, such as Enter"),
                     ToolParam(name="selector", description="CSS selector, or none for the page", required=False)]),
)


class PolicyError(RuntimeError):
    """The policy chose something the environment cannot run."""


@dataclass(frozen=True)
class Submit:
    """The model says the task is done."""


@dataclass(frozen=True)
class Turn:
    step: GoldenStep
    result: StepResult


class Policy(Protocol):
    def next_step(self, history: Sequence[Turn]) -> GoldenStep | Submit: ...


def action_tools(profile: EnvironmentProfile) -> list[ToolSpec]:
    """The tools a model may call in this environment, `submit` included."""
    if profile.family == "cli":
        tools = list(_CLI_TOOLS)
    elif profile.family == "browser":
        tools = list(_BROWSER_TOOLS)
    else:
        tools = [
            ToolSpec(name=tool.name, description=tool.description, params=[_param(hint) for hint in tool.params])
            for tool in profile.tools
        ]
    return [*tools, _SUBMIT_TOOL]


def _param(hint: str) -> ToolParam:
    """A profile hint, "name: type" or "name: a|b|c", as a tool parameter.

    Hints do not say whether a parameter is required, so none is marked so.
    """
    name, _, kind = (part.strip() for part in hint.partition(":"))
    if "|" in kind:
        return ToolParam(name=name, description=f"one of {', '.join(kind.split('|'))}", required=False)
    return ToolParam(name=name, type=kind if kind in _JSON_TYPES else "string", required=False)


class AgentPolicy:
    def __init__(self, agent: AgentAdapter, tools: Sequence[ToolSpec]) -> None:
        self._agent = agent
        self._names = frozenset(tool.name for tool in tools)

    def next_step(self, history: Sequence[Turn]) -> GoldenStep | Submit:
        action = dict(self._agent.act({"history": [_seen(turn) for turn in history]}, self._names))
        name = action.pop("type", None)
        if name == SUBMIT:
            return Submit()
        if name not in self._names:
            raise PolicyError(f"the model called {name!r}, which this environment does not have")
        return GoldenStep(tool=name, args=action)


def agent_policy(agent_id: str, task: SyntheticTask, profile: EnvironmentProfile) -> AgentPolicy:
    """A policy for one task, driving the agent `agent_id` names, such as ollama:qwen3.8:27b."""
    tools = action_tools(profile)
    agent = make_agent(
        agent_id,
        task=Task(id=task.id, objective=task.objective),
        tool_specs=tools,
        prompt_template=ForgeAgentPromptTemplate(),
    )
    return AgentPolicy(agent, tools)


def _seen(turn: Turn) -> dict:
    seen = {"tool": turn.step.tool, "args": turn.step.args, "ok": turn.result.ok}
    if turn.result.ok:
        seen["output"] = _cut(turn.result.output)
    else:
        seen["error"] = _cut(turn.result.error)
    return seen


def _cut(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    return f"{text[:OUTPUT_LIMIT]}\n[{len(text) - OUTPUT_LIMIT} more characters cut]"
