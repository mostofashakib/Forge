from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, ClassVar, Literal

from pydantic import BaseModel

from forge.runtime.errors import (
    BrowserContractViolation,
    ComputerContractViolation,
    InvalidActionError,
    MCPContractViolation,
    ORPCContractViolation,
    RESTContractViolation,
    ToolContractViolation,
)
from forge.runtime.snapshot import ToolParam, ToolSpec

_PARAM_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _validate_params(params: list[ToolParam], values: dict, owner: str) -> str | None:
    """Check declared params against a dict of supplied values.

    Shared by every schema whose actions carry named parameters (tool, MCP,
    REST, oRPC), so the required/typing rules stay identical across modalities.
    """
    if not isinstance(values, dict):
        return f"'{owner}' arguments must be an object"
    for param in params:
        if param.required and param.name not in values:
            return f"{owner} requires param '{param.name}'"
        expected = _PARAM_TYPES.get(param.type)
        if param.name in values and expected:
            value = values[param.name]
            # bool subclasses int, but True is not a valid integer/number param.
            wrong_type = not isinstance(value, expected) or (
                isinstance(value, bool) and param.type in ("integer", "number")
            )
            if wrong_type:
                return f"param '{param.name}' of {owner} must be {param.type}"
    return None


class ToolUseSchema(BaseModel):
    """Contract for tool interaction — the API endpoints / functions of the
    environment an agent may call, their parameters, and what counts as a
    well-formed call."""

    tools: list[ToolSpec]

    def tool_names(self) -> list[str]:
        return sorted(spec.name for spec in self.tools)

    def tool_specs(self) -> list[ToolSpec]:
        """The env's tool-call actions, as tool-surface entries."""
        return list(self.tools)

    def validate_action(self, action: dict) -> str | None:
        if not isinstance(action, dict) or "type" not in action:
            return "tool call must be a dict with a 'type' field"
        spec = next((s for s in self.tools if s.name == action["type"]), None)
        if spec is None:
            return f"unknown tool '{action['type']}'; available: {self.tool_names()}"
        return _validate_params(spec.params, action, f"'{spec.name}'")


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

    def tool_specs(self) -> list[ToolSpec]:
        """The OS primitives available on this machine, as tool-surface entries."""
        params = {"exec": [ToolParam(name="command", type="string")]}
        return [
            ToolSpec(name=a, description=f"{self.os} OS primitive", params=params.get(a, []))
            for a in self.actions
        ]


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

    def tool_specs(self) -> list[ToolSpec]:
        """The page primitives available in this viewport, as tool-surface entries."""
        return [
            ToolSpec(name=a, description=f"browser primitive ({self.display_width}x{self.display_height})")
            for a in self.actions
        ]


class MCPUseSchema(BaseModel):
    """Contract for MCP interaction — the Model Context Protocol tools a server
    exposes to the agent, their input parameters, and what counts as a
    well-formed ``tools/call``.

    Action shape: ``{"tool": "<name>", "arguments": {...}}``.
    """

    server: str = "default"
    tools: list[ToolSpec]

    def tool_names(self) -> list[str]:
        return sorted(tool.name for tool in self.tools)

    def tool_specs(self) -> list[ToolSpec]:
        return list(self.tools)

    def validate_action(self, action: dict) -> str | None:
        if not isinstance(action, dict) or "tool" not in action:
            return "MCP call must be a dict with a 'tool' field"
        tool = next((t for t in self.tools if t.name == action["tool"]), None)
        if tool is None:
            return f"unknown MCP tool '{action['tool']}'; available: {self.tool_names()}"
        return _validate_params(tool.params, action.get("arguments", {}), f"MCP tool '{tool.name}'")


class RESTEndpoint(BaseModel):
    """One REST endpoint the environment exposes: an HTTP method + path and the
    parameters its request accepts."""

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    description: str = ""
    params: list[ToolParam] = []

    def signature(self) -> str:
        return f"{self.method} {self.path}"


class RESTUseSchema(BaseModel):
    """Contract for REST interaction — the HTTP endpoints an agent may call.

    Action shape: ``{"method": "POST", "path": "/tickets", "input": {...}}``.
    """

    base_url: str = ""
    endpoints: list[RESTEndpoint]

    def signatures(self) -> list[str]:
        return sorted(e.signature() for e in self.endpoints)

    def tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=e.signature(), description=e.description, params=list(e.params))
            for e in self.endpoints
        ]

    def validate_action(self, action: dict) -> str | None:
        if not isinstance(action, dict) or "method" not in action or "path" not in action:
            return "REST call must be a dict with 'method' and 'path' fields"
        method = str(action["method"]).upper()
        endpoint = next(
            (e for e in self.endpoints if e.method == method and e.path == action["path"]), None
        )
        if endpoint is None:
            return (
                f"unknown REST endpoint '{method} {action['path']}'; "
                f"available: {self.signatures()}"
            )
        return _validate_params(
            endpoint.params, action.get("input", {}), f"REST endpoint '{endpoint.signature()}'"
        )


class ORPCProcedure(BaseModel):
    """One oRPC procedure the environment exposes: a (dotted) name and the input
    parameters it accepts."""

    name: str
    description: str = ""
    params: list[ToolParam] = []


class ORPCUseSchema(BaseModel):
    """Contract for oRPC interaction — the typed RPC procedures an agent may
    invoke, their input contract, and what counts as a well-formed call.

    Action shape: ``{"procedure": "tickets.create", "input": {...}}``.
    """

    procedures: list[ORPCProcedure]

    def procedure_names(self) -> list[str]:
        return sorted(p.name for p in self.procedures)

    def tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=p.name, description=p.description, params=list(p.params))
            for p in self.procedures
        ]

    def validate_action(self, action: dict) -> str | None:
        if not isinstance(action, dict) or "procedure" not in action:
            return "oRPC call must be a dict with a 'procedure' field"
        proc = next((p for p in self.procedures if p.name == action["procedure"]), None)
        if proc is None:
            return (
                f"unknown oRPC procedure '{action['procedure']}'; "
                f"available: {self.procedure_names()}"
            )
        return _validate_params(proc.params, action.get("input", {}), f"oRPC procedure '{proc.name}'")


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


@dataclass
class MCPUse(Capability):
    """MCP capability: validated calls to an MCP server's tools."""

    schema: MCPUseSchema
    name = "mcp_use"
    violation = MCPContractViolation


@dataclass
class RESTUse(Capability):
    """REST capability: validated calls to the environment's HTTP endpoints."""

    schema: RESTUseSchema
    name = "rest_use"
    violation = RESTContractViolation


@dataclass
class ORPCUse(Capability):
    """oRPC capability: validated calls to the environment's typed RPC procedures."""

    schema: ORPCUseSchema
    name = "orpc_use"
    violation = ORPCContractViolation
