"""Agent implementations."""

from typing import Any

__all__ = ["SlackExternalAgent", "TaskManagerExternalAgent"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        if name == "SlackExternalAgent":
            from fleet.agents.rl_agent import SlackExternalAgent
            return SlackExternalAgent
        elif name == "TaskManagerExternalAgent":
            from fleet.agents.rl_agent import TaskManagerExternalAgent
            return TaskManagerExternalAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

