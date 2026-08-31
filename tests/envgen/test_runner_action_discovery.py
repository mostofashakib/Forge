"""The runner discovers actions through the shared OpenAPIToolProvider.

Action discovery used to be private to ContainerEpisodeRunner. It now lives in
`forge.runtime.tools.OpenAPIToolProvider`, and these tests pin that moving it
changed nothing the runner depends on: the same manifest, the same caching, and
the same empty-list-not-an-exception behavior when the app is unreachable.
"""
from __future__ import annotations

import httpx

from forge.contracts import ToolProvider
from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig

_SCHEMA = {
    "paths": {
        "/close_ticket": {"post": {"summary": "Close a ticket"}},
        "/forge/reset": {"post": {"summary": "Forge internals"}},
        "/ui": {"post": {"summary": "The app"}},
        "/tickets": {"get": {"summary": "List"}},
    },
    "components": {"schemas": {}},
}


def _runner(*, fail: bool = False) -> tuple[ContainerEpisodeRunner, list[str]]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if fail:
            raise httpx.ConnectError("container is not up")
        return httpx.Response(200, json=_SCHEMA)

    runner = ContainerEpisodeRunner(
        EpisodeConfig(base_url="http://app", objective="close the ticket")
    )
    runner._http = httpx.Client(
        base_url="http://app", transport=httpx.MockTransport(handler)
    )
    return runner, calls


def test_the_runner_exposes_a_tool_provider():
    runner, _ = _runner()

    assert isinstance(runner.tool_provider, ToolProvider)


def test_discovery_returns_the_same_manifest_the_runner_always_used():
    runner, _ = _runner()

    actions = runner._discover_actions()

    assert actions == [{
        "endpoint": "/close_ticket",
        "description": "Close a ticket",
        "request_schema": {},
    }]


def test_discovery_still_excludes_forge_internals_and_the_ui():
    # Negative case: the runner drove these exclusions before the move.
    runner, _ = _runner()

    endpoints = [action["endpoint"] for action in runner._discover_actions()]

    assert "/forge/reset" not in endpoints
    assert "/ui" not in endpoints
    # False-positive guard: /tickets has only a GET, so a provider that took
    # every path rather than every POST would still pass the two assertions
    # above.
    assert "/tickets" not in endpoints


def test_discovery_is_still_cached_across_calls():
    runner, calls = _runner()

    runner._discover_actions()
    runner._discover_actions()

    assert calls.count("/openapi.json") == 1


def test_an_unreachable_app_still_yields_an_empty_list_rather_than_raising():
    runner, _ = _runner(fail=True)

    assert runner._discover_actions() == []
