"""What is real in one environment: the facts every factory step works from.

Built by the backend from the environment's compiled input, its running
container, or its type. The steps never read an environment any other way,
so a task can only name tools and state that exist.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from forge.taskfactory.schemas import EnvFamily

# What a CLI task can rely on, shown to the writer and validator.
CLI_NOTES = (
    "Ubuntu 22.04 shell, run as root, no network. Coreutils, bash, grep, sed, awk, "
    "find, tar and gzip are installed. Python and package managers are not usable "
    "without the network. The clock is frozen at 2023-11-14 22:13:20 UTC."
)
BROWSER_NOTES = (
    "Chromium with no internet. Each task supplies its own static pages, served at "
    "http://task.local/<path>. Pages may use inline JavaScript but no external "
    "resources. Golden steps: goto {path}, click {selector}, fill {selector, value}, "
    "select {selector, value}, press {selector, key}, check {selector}."
)
_SAMPLE_CHARS = 6000


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str = ""
    params: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentProfile:
    env_name: str
    env_type: str
    family: EnvFamily
    tools: tuple[ToolInfo, ...] = ()
    # The reset state: seed rows merge into it and checks read it.
    state_sample: dict | None = None
    notes: str = ""
    _patterns: tuple[re.Pattern, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_patterns", tuple(_tool_pattern(t.name) for t in self.tools))

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def has_tool(self, name: str) -> bool:
        """True when `name` is a tool, or fills one templated endpoint like `/a/{id}`."""
        return any(pattern.fullmatch(name) for pattern in self._patterns)

    def prompt_view(self) -> str:
        """The profile as the writer and validator read it."""
        lines = [f"Environment: {self.env_name} (type {self.env_type})"]
        if self.notes:
            lines.append(self.notes)
        if self.tools:
            lines.append("Tools (use these names exactly):")
            for tool in self.tools:
                params = f"({', '.join(tool.params)})" if tool.params else "()"
                desc = f" - {tool.description}" if tool.description else ""
                lines.append(f"  {tool.name} {params}{desc}")
        if self.state_sample is not None:
            lines.append(
                "Reset state (seed rows merge into these collections by id, and checks read "
                f"this same shape):\n{_sample(self.state_sample)}"
            )
        return "\n".join(lines)


def _tool_pattern(name: str) -> re.Pattern:
    parts = re.split(r"(\{[^/{}]+\})", name)
    return re.compile("".join("[^/]+" if p.startswith("{") else re.escape(p) for p in parts))


def _sample(state: dict) -> str:
    text = json.dumps(state, sort_keys=True, default=str)
    return text if len(text) <= _SAMPLE_CHARS else text[:_SAMPLE_CHARS] + " …(truncated)"
