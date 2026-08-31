"""The three concrete ToolProviders.

`OpenAPIToolProvider` carries the discovery logic that used to be private to
`ContainerEpisodeRunner`, so its tests pin the behavior that runner depended
on: the /forge/* and /ui exclusions, POST-only, $ref resolution, caching, and
returning an empty manifest rather than raising when the app is unreachable.
"""
from __future__ import annotations

import httpx
import pytest

from forge.contracts import ToolParam, ToolProvider, ToolSpec
from forge.runtime.interaction import ComputerUse, ComputerUseSchema, RESTUse, RESTUseSchema
from forge.runtime.tools import CapabilityToolProvider, OpenAPIToolProvider, SpecToolProvider

_SCHEMA = {
    "paths": {
        "/close_ticket": {
            "post": {
                "summary": "Close a ticket",
                "requestBody": {
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/CloseTicket"}}
                    }
                },
            }
        },
        "/notify": {
            "post": {
                "operationId": "notify_customer",
                "requestBody": {
                    "content": {"application/json": {"schema": {"type": "object"}}}
                },
            }
        },
        "/tickets": {"get": {"summary": "List tickets"}},
        "/forge/state": {"post": {"summary": "Forge internals"}},
        "/ui": {"post": {"summary": "The browsable app"}},
    },
    "components": {
        "schemas": {
            "CloseTicket": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string", "description": "Which ticket"}},
                "required": ["ticket_id"],
            }
        }
    },
}


def _client(schema: dict | None = None, *, fail: bool = False) -> httpx.Client:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if fail:
            raise httpx.ConnectError("container is not up")
        return httpx.Response(200, json=schema)

    client = httpx.Client(
        base_url="http://app", transport=httpx.MockTransport(handler)
    )
    client.forge_calls = calls  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# SpecToolProvider
# ---------------------------------------------------------------------------

def test_a_static_provider_returns_what_it_was_given():
    tools = [ToolSpec(name="close_ticket"), ToolSpec(name="reply")]

    provider = SpecToolProvider(tools)

    assert isinstance(provider, ToolProvider)
    assert [tool.name for tool in provider.tools()] == ["close_ticket", "reply"]


def test_a_static_provider_does_not_alias_the_caller_list():
    # False-positive guard: mutating the caller's list must not silently
    # change the environment's tool surface after construction.
    tools = [ToolSpec(name="close_ticket")]
    provider = SpecToolProvider(tools)

    tools.append(ToolSpec(name="smuggled"))

    assert [tool.name for tool in provider.tools()] == ["close_ticket"]


def test_an_empty_static_provider_is_legal():
    # Negative case: an environment may legitimately expose no tools.
    assert list(SpecToolProvider([]).tools()) == []


# ---------------------------------------------------------------------------
# CapabilityToolProvider
# ---------------------------------------------------------------------------

def test_a_capability_provider_exposes_its_capabilitys_tools():
    capability = ComputerUse(schema=ComputerUseSchema(), executor=lambda action: None)

    provider = CapabilityToolProvider([capability])

    assert isinstance(provider, ToolProvider)
    assert [tool.name for tool in provider.tools()] == ["exec", "screenshot", "noop"]


def test_a_capability_provider_concatenates_several_capabilities():
    rest = RESTUse(
        schema=RESTUseSchema(
            endpoints=[
                {
                    "method": "POST",
                    "path": "/tickets",
                    "description": "Create a ticket",
                    "params": [ToolParam(name="subject")],
                }
            ]
        ),
        executor=lambda action: None,
    )
    computer = ComputerUse(
        schema=ComputerUseSchema(actions=["noop"]), executor=lambda action: None
    )

    names = [tool.name for tool in CapabilityToolProvider([rest, computer]).tools()]

    assert names == ["POST /tickets", "noop"]


def test_a_capability_without_a_tool_surface_is_rejected():
    # Negative case: silently returning no tools for a capability that cannot
    # describe itself would hide the misconfiguration until the agent had
    # nothing to call.
    class Opaque:
        pass

    class Bare:
        schema = Opaque()

    with pytest.raises(TypeError, match="tool_specs"):
        CapabilityToolProvider([Bare()])


# ---------------------------------------------------------------------------
# OpenAPIToolProvider
# ---------------------------------------------------------------------------

def test_openapi_discovery_finds_the_declared_post_endpoints():
    provider = OpenAPIToolProvider(_client(_SCHEMA))

    manifest = provider.action_manifest()

    assert [entry["endpoint"] for entry in manifest] == ["/close_ticket", "/notify"]


def test_openapi_discovery_excludes_forge_internals_and_the_ui():
    # Negative case: these paths exist in the schema and must not be offered.
    endpoints = [
        entry["endpoint"] for entry in OpenAPIToolProvider(_client(_SCHEMA)).action_manifest()
    ]

    assert "/forge/state" not in endpoints
    assert "/ui" not in endpoints


def test_openapi_discovery_ignores_endpoints_without_a_post():
    # False-positive guard: /tickets has a GET, so a provider that collected
    # every path rather than every POST would pass the exclusion test above.
    endpoints = [
        entry["endpoint"] for entry in OpenAPIToolProvider(_client(_SCHEMA)).action_manifest()
    ]

    assert "/tickets" not in endpoints


def test_openapi_discovery_resolves_a_referenced_request_schema():
    manifest = OpenAPIToolProvider(_client(_SCHEMA)).action_manifest()

    close_ticket = next(e for e in manifest if e["endpoint"] == "/close_ticket")
    assert close_ticket["request_schema"]["properties"]["ticket_id"]["type"] == "string"


def test_openapi_discovery_falls_back_to_the_operation_id_for_a_description():
    manifest = OpenAPIToolProvider(_client(_SCHEMA)).action_manifest()

    assert next(e for e in manifest if e["endpoint"] == "/notify")["description"] == (
        "notify_customer"
    )


def test_openapi_discovery_is_cached_after_the_first_call():
    client = _client(_SCHEMA)
    provider = OpenAPIToolProvider(client)

    provider.action_manifest()
    provider.action_manifest()

    assert len(client.forge_calls) == 1  # type: ignore[attr-defined]


def test_an_unreachable_app_yields_an_empty_manifest_rather_than_raising():
    # The runner treats discovery as best-effort: a container that is not up
    # must cost an empty action list, never an exception out of the episode.
    provider = OpenAPIToolProvider(_client(fail=True))

    assert provider.action_manifest() == []
    assert list(provider.tools()) == []


def test_openapi_tools_expose_the_request_schema_as_tool_parameters():
    provider = OpenAPIToolProvider(_client(_SCHEMA))

    tools = list(provider.tools())

    close_ticket = next(tool for tool in tools if tool.name == "/close_ticket")
    assert close_ticket.description == "Close a ticket"
    assert [param.name for param in close_ticket.params] == ["ticket_id"]
    assert close_ticket.params[0].description == "Which ticket"
    assert close_ticket.params[0].required is True


def test_an_optional_request_field_is_not_marked_required():
    # False-positive guard: marking everything required would still satisfy
    # the assertion above, and would misdescribe the tool to the model.
    schema = {
        "paths": {
            "/notify": {
                "post": {
                    "summary": "Notify",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "who": {"type": "string"},
                                        "cc": {"type": "string"},
                                    },
                                    "required": ["who"],
                                }
                            }
                        }
                    },
                }
            }
        }
    }
    tool = next(iter(OpenAPIToolProvider(_client(schema)).tools()))

    required = {param.name: param.required for param in tool.params}
    assert required == {"who": True, "cc": False}
