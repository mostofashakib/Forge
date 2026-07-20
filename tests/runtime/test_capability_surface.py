"""ForgeEnv exposes every attached interaction modality through the tool surface."""
from __future__ import annotations

import copy

import pytest

from forge.runtime.env_builder import EnvBuilder
from forge.runtime.errors import RESTContractViolation
from forge.runtime.interaction import (
    MCPUseSchema,
    ORPCProcedure,
    ORPCUseSchema,
    RESTEndpoint,
    RESTUseSchema,
)
from forge.runtime.snapshot import ToolParam, ToolSpec
from forge.runtime.transition import TransitionResult


class CounterFactory:
    def create(self, ctx, options):
        return {"counter": {"c_0": {"id": "c_0", "value": 0}}}


def _increment(state, action, ctx):
    new_state = copy.deepcopy(state)
    new_state["counter"]["c_0"]["value"] += 1
    return TransitionResult(state=new_state, events=[])


def _builder() -> EnvBuilder:
    return (
        EnvBuilder("iface_env", domain="test", max_steps=10)
        .with_initial_state(CounterFactory())
        .with_transition("increment", _increment)
    )


def _mcp() -> MCPUseSchema:
    return MCPUseSchema(tools=[ToolSpec(name="search", params=[ToolParam(name="q")])])


def _rest() -> RESTUseSchema:
    return RESTUseSchema(endpoints=[RESTEndpoint(method="GET", path="/items")])


def _orpc() -> ORPCUseSchema:
    return ORPCUseSchema(procedures=[ORPCProcedure(name="items.list")])


def test_env_without_extra_modalities_has_only_tool_use():
    env = _builder().build(verify=False)
    assert env.capabilities() == ["tool_use"]
    assert list(env.capability_surface()) == ["tool_use"]
    assert env.mcp_use is None and env.rest_use is None and env.orpc_use is None


def test_builder_attaches_mcp_rest_and_orpc():
    env = (
        _builder()
        .with_mcp_use(executor=lambda a: None, schema=_mcp())
        .with_rest_use(executor=lambda a: None, schema=_rest())
        .with_orpc_use(executor=lambda a: None, schema=_orpc())
        .build(verify=False)
    )
    assert env.capabilities() == ["tool_use", "mcp_use", "rest_use", "orpc_use"]


def test_capability_surface_exposes_every_modalitys_actions():
    env = (
        _builder()
        .with_mcp_use(executor=lambda a: None, schema=_mcp())
        .with_rest_use(executor=lambda a: None, schema=_rest())
        .with_orpc_use(executor=lambda a: None, schema=_orpc())
        .with_computer_use(executor=lambda a: None)
        .with_browser_use(executor=lambda a: None)
        .build(verify=False)
    )
    surface = env.capability_surface()
    assert set(surface) == {
        "tool_use", "mcp_use", "rest_use", "orpc_use", "computer_use", "browser_use",
    }
    assert [s.name for s in surface["tool_use"]] == ["increment"]
    assert [s.name for s in surface["mcp_use"]] == ["search"]
    assert [s.name for s in surface["rest_use"]] == ["GET /items"]
    assert [s.name for s in surface["orpc_use"]] == ["items.list"]
    assert "exec" in {s.name for s in surface["computer_use"]}
    assert "click" in {s.name for s in surface["browser_use"]}


def test_attached_capability_validates_and_executes():
    calls = []
    env = _builder().with_rest_use(executor=calls.append, schema=_rest()).build(verify=False)
    env.rest_use.execute({"method": "GET", "path": "/items"})
    assert calls == [{"method": "GET", "path": "/items"}]
    with pytest.raises(RESTContractViolation):
        env.rest_use.execute({"method": "DELETE", "path": "/items"})


def test_only_declared_modalities_appear():
    # An env that needs only MCP does not advertise REST/oRPC/computer/browser.
    env = _builder().with_mcp_use(executor=lambda a: None, schema=_mcp()).build(verify=False)
    assert env.capabilities() == ["tool_use", "mcp_use"]
    assert set(env.capability_surface()) == {"tool_use", "mcp_use"}
