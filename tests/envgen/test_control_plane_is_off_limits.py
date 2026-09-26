"""An agent's action can never reach the environment's control plane.

`/forge/reset`, `/forge/restore-state`, `/forge/snapshot`, and friends decide
the state the grader reads. If an agent could call them, it could write the
"solved" state directly. Agent and persona actions go through the execution
backend, which refuses those targets without contacting the app.
"""
from __future__ import annotations

import httpx
import pytest

from forge.envgen.container_env_base import ContainerEnvBase


def _env(seen: list[str]) -> ContainerEnvBase:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if request.url.path == "/forge/state":
            return httpx.Response(200, json={"n": 0})
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ContainerEnvBase("http://app", client=client)


@pytest.mark.parametrize("target", [
    "/forge/restore-state", "/forge/reset", "/forge/snapshot", "/forge/restore/baseline",
    "forge/restore-state", "/ui",
])
def test_an_action_aimed_at_the_control_plane_is_refused_without_reaching_the_app(target):
    seen: list[str] = []
    env = _env(seen)

    _obs, reward, _term, _trunc, info = env.step({"type": target})

    assert not any(call.startswith("POST") for call in seen)
    assert info["status_code"] == 403
    assert reward == 0.0


def test_a_domain_action_still_reaches_the_app():
    # False-positive guard: only the control plane is off limits.
    seen: list[str] = []
    env = _env(seen)

    _obs, _reward, _term, _trunc, info = env.step({"type": "archive_email"})

    assert "POST /archive_email" in seen
    assert info["status_code"] == 200
