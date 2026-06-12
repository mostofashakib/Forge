from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, ClassVar, Literal

from pydantic import BaseModel

from forge.runtime.errors import (
    BrowserContractViolation,
    ComputerContractViolation,
    InvalidActionError,
    ToolContractViolation,
)
from forge.runtime.snapshot import ToolSpec

_PARAM_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


class ToolUseSchema(BaseModel):
    """Contract for tool interaction — the API endpoints / functions of the
    environment an agent may call, their parameters, and what counts as a
    well-formed call."""

    tools: list[ToolSpec]

    def tool_names(self) -> list[str]:
        return sorted(spec.name for spec in self.tools)

    def validate_action(self, action: dict) -> str | None:
        if not isinstance(action, dict) or "type" not in action:
            return "tool call must be a dict with a 'type' field"
        spec = next((s for s in self.tools if s.name == action["type"]), None)
        if spec is None:
            return f"unknown tool '{action['type']}'; available: {self.tool_names()}"
        for param in spec.params:
            if param.required and param.name not in action:
                return f"tool '{spec.name}' requires param '{param.name}'"
            expected = _PARAM_TYPES.get(param.type)
            if param.name in action and expected:
                value = action[param.name]
                # bool subclasses int, but True is not a valid integer/number param
                wrong_type = not isinstance(value, expected) or (
                    isinstance(value, bool) and param.type in ("integer", "number")
                )
                if wrong_type:
                    return f"param '{param.name}' of '{spec.name}' must be {param.type}"
        return None


class ComputerUseSchema(BaseModel):
    """Contract for VM / OS interaction — the operating-system primitives an
    agent may use inside the machine (shell, screenshots)."""

    os: Literal["linux", "macos", "windows"] = "linux"
    actions: list[str] = ["exec", "screenshot", "noop"]

    def validate_action(self, action: dict) -> str | None:
        if not isinstance(action, dict) or "action_type" not in action:
            return "computer action must be a dict with an 'action_type' field"
        action_type = action["action_type"]
        if action_type not in self.actions:
            return f"unknown primitive '{action_type}'; available: {self.actions}"
        if action_type == "exec":
            command = action.get("command")
            if not isinstance(command, str) or not command.strip():
                return "exec requires a non-empty string 'command'"
        return None


class BrowserUseSchema(BaseModel):
    """Contract for browser interaction — the page primitives an agent may
    use and the viewport bounds they must stay within."""

    display_width: int = 1280
    display_height: int = 720
    actions: list[str] = ["click", "type", "press", "navigate", "scroll", "noop"]

    def validate_action(self, action: dict) -> str | None:
        if not isinstance(action, dict) or "action_type" not in action:
            return "browser action must be a dict with an 'action_type' field"
        action_type = action["action_type"]
        if action_type not in self.actions:
            return f"unknown primitive '{action_type}'; available: {self.actions}"
        if action_type == "click":
            x, y = action.get("x"), action.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return "click requires numeric 'x' and 'y'"
            if not (0 <= x < self.display_width and 0 <= y < self.display_height):
                return (
                    f"click ({x}, {y}) outside display "
                    f"{self.display_width}x{self.display_height}"
                )
        elif action_type == "type":
            if not isinstance(action.get("text"), str):
                return "type requires a string 'text'"
        elif action_type == "press":
            if not isinstance(action.get("key"), str):
                return "press requires a string 'key'"
        elif action_type == "navigate":
            if not action.get("url"):
                return "navigate requires a non-empty 'url'"
        elif action_type == "scroll":
            for axis in ("delta_x", "delta_y"):
                if axis in action and not isinstance(action[axis], (int, float)):
                    return f"scroll '{axis}' must be numeric"
        return None


@dataclass
class Capability:
    """A validated interaction surface handed to an agent: actions are checked
    against the schema's contract before the executor ever runs."""

    schema: object
    executor: Callable[[dict], object]
    name = "capability"
    violation: ClassVar[type[InvalidActionError]] = InvalidActionError

    def execute(self, action: dict):
        error = self.schema.validate_action(action)
        if error:
            raise self.violation(error)
        return self.executor(action)


@dataclass
class ToolUse(Capability):
    """Tool-calling capability: validated calls to the environment's API
    endpoints / functions."""

    schema: ToolUseSchema
    name = "tool_use"
    violation = ToolContractViolation


@dataclass
class ComputerUse(Capability):
    """OS capability: validated execution of VM/OS primitives
    (linux/macOS/Windows)."""

    schema: ComputerUseSchema
    name = "computer_use"
    violation = ComputerContractViolation


@dataclass
class BrowserUse(Capability):
    """Browser capability: validated execution of page primitives against
    a viewport."""

    schema: BrowserUseSchema
    name = "browser_use"
    violation = BrowserContractViolation
