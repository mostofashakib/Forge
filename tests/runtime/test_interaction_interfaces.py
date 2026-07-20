"""Contracts for the agent interaction modalities: MCP, REST, oRPC.

Computer use and browser use are covered in test_interaction.py; this file
covers the three additional interfaces and the uniform tool-surface enumeration
every schema exposes.
"""
from __future__ import annotations

import pytest

from forge.runtime.errors import (
    MCPContractViolation,
    ORPCContractViolation,
    RESTContractViolation,
)
from forge.runtime.interaction import (
    BrowserUseSchema,
    ComputerUseSchema,
    MCPUse,
    MCPUseSchema,
    ORPCProcedure,
    ORPCUse,
    ORPCUseSchema,
    RESTEndpoint,
    RESTUse,
    RESTUseSchema,
    ToolUseSchema,
)
from forge.runtime.snapshot import ToolParam, ToolSpec


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------

def _mcp_schema() -> MCPUseSchema:
    return MCPUseSchema(server="tickets", tools=[
        ToolSpec(name="search_tickets", params=[ToolParam(name="query", type="string")]),
        ToolSpec(name="list_agents", params=[]),
    ])


def test_mcp_accepts_well_formed_call():
    schema = _mcp_schema()
    assert schema.validate_action(
        {"tool": "search_tickets", "arguments": {"query": "billing"}}
    ) is None


def test_mcp_rejects_missing_tool_field():
    assert "tool" in _mcp_schema().validate_action({"arguments": {}})


def test_mcp_rejects_unknown_tool():
    err = _mcp_schema().validate_action({"tool": "delete_db", "arguments": {}})
    assert "unknown MCP tool" in err


def test_mcp_rejects_missing_required_argument():
    err = _mcp_schema().validate_action({"tool": "search_tickets", "arguments": {}})
    assert "query" in err


def test_mcp_rejects_wrong_argument_type():
    err = _mcp_schema().validate_action(
        {"tool": "search_tickets", "arguments": {"query": 5}}
    )
    assert "string" in err


def test_mcp_capability_executes_and_enforces_contract():
    calls = []
    cap = MCPUse(schema=_mcp_schema(), executor=calls.append)
    assert cap.name == "mcp_use"
    cap.execute({"tool": "list_agents", "arguments": {}})
    assert calls == [{"tool": "list_agents", "arguments": {}}]
    with pytest.raises(MCPContractViolation):
        cap.execute({"tool": "nope"})


def test_mcp_tool_specs_enumerate_the_contract():
    specs = _mcp_schema().tool_specs()
    assert {s.name for s in specs} == {"search_tickets", "list_agents"}
    assert all(isinstance(s, ToolSpec) for s in specs)


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

def _rest_schema() -> RESTUseSchema:
    return RESTUseSchema(endpoints=[
        RESTEndpoint(method="GET", path="/tickets", params=[]),
        RESTEndpoint(
            method="POST", path="/tickets",
            params=[ToolParam(name="title", type="string")],
        ),
    ])


def test_rest_accepts_declared_endpoint():
    schema = _rest_schema()
    assert schema.validate_action(
        {"method": "POST", "path": "/tickets", "input": {"title": "help"}}
    ) is None


def test_rest_is_case_insensitive_on_method():
    assert _rest_schema().validate_action({"method": "get", "path": "/tickets"}) is None


def test_rest_rejects_unknown_endpoint():
    err = _rest_schema().validate_action({"method": "DELETE", "path": "/tickets"})
    assert "unknown REST endpoint" in err


def test_rest_rejects_missing_required_param():
    err = _rest_schema().validate_action({"method": "POST", "path": "/tickets", "input": {}})
    assert "title" in err


def test_rest_capability_executes_and_enforces_contract():
    calls = []
    cap = RESTUse(schema=_rest_schema(), executor=calls.append)
    assert cap.name == "rest_use"
    cap.execute({"method": "GET", "path": "/tickets"})
    assert calls == [{"method": "GET", "path": "/tickets"}]
    with pytest.raises(RESTContractViolation):
        cap.execute({"method": "PUT", "path": "/tickets"})


def test_rest_tool_specs_name_by_method_and_path():
    names = {s.name for s in _rest_schema().tool_specs()}
    assert names == {"GET /tickets", "POST /tickets"}


# ---------------------------------------------------------------------------
# oRPC
# ---------------------------------------------------------------------------

def _orpc_schema() -> ORPCUseSchema:
    return ORPCUseSchema(procedures=[
        ORPCProcedure(name="tickets.create", params=[ToolParam(name="title", type="string")]),
        ORPCProcedure(name="tickets.list", params=[]),
    ])


def test_orpc_accepts_declared_procedure():
    assert _orpc_schema().validate_action(
        {"procedure": "tickets.create", "input": {"title": "x"}}
    ) is None


def test_orpc_rejects_unknown_procedure():
    err = _orpc_schema().validate_action({"procedure": "tickets.nuke"})
    assert "unknown oRPC procedure" in err


def test_orpc_rejects_missing_required_input():
    err = _orpc_schema().validate_action({"procedure": "tickets.create", "input": {}})
    assert "title" in err


def test_orpc_capability_executes_and_enforces_contract():
    calls = []
    cap = ORPCUse(schema=_orpc_schema(), executor=calls.append)
    assert cap.name == "orpc_use"
    cap.execute({"procedure": "tickets.list", "input": {}})
    assert calls == [{"procedure": "tickets.list", "input": {}}]
    with pytest.raises(ORPCContractViolation):
        cap.execute({"procedure": "tickets.nuke"})


def test_orpc_tool_specs_use_procedure_names():
    names = {s.name for s in _orpc_schema().tool_specs()}
    assert names == {"tickets.create", "tickets.list"}


# ---------------------------------------------------------------------------
# Uniform tool_specs() across all modalities
# ---------------------------------------------------------------------------

def test_every_schema_exposes_tool_specs():
    schemas = [
        ToolUseSchema(tools=[ToolSpec(name="a")]),
        ComputerUseSchema(),
        BrowserUseSchema(),
        _mcp_schema(),
        _rest_schema(),
        _orpc_schema(),
    ]
    for schema in schemas:
        specs = schema.tool_specs()
        assert isinstance(specs, list)
        assert specs, f"{type(schema).__name__} produced no tool specs"
        assert all(isinstance(s, ToolSpec) for s in specs)
