"""The container env's HTTP collaborators run through its own RestTransport.

`RestTransport` was hardened for a timeout hole and a JSON-decode gap, but the
two collaborators held their own `httpx.Client`, so that hardening protected
nothing that actually ran. They now go through the transport.

`RestTransport.call()` reports failures in-band and never raises, while this
family's callers have always seen exceptions. So the conversion back to a raise
happens at the boundary, and the bulk of this file pins that today's raise /
no-raise behavior is unchanged — that mapping is the whole risk of the change.
"""
from __future__ import annotations

import httpx
import pytest

from forge.contracts import Action, Transport
from forge.envgen.container_env_base import ContainerEnvBase, ContainerTransportError


def _env(handler, **kwargs) -> ContainerEnvBase:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ContainerEnvBase("http://app", client=client, **kwargs)


def _ok(state: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/forge/state":
            return httpx.Response(200, json=state or {"n": 0})
        return httpx.Response(200, json={})
    return handler


# ---------------------------------------------------------------------------
# The routing itself
# ---------------------------------------------------------------------------

def test_the_collaborators_share_the_envs_transport():
    env = _env(_ok())

    assert isinstance(env.transport, Transport)
    assert env.backend._transport is env.transport
    assert env.initial_state._transport is env.transport


def test_reset_and_step_still_work_through_the_transport():
    # False-positive guard: sharing the object proves nothing if the happy
    # path no longer functions.
    env = _env(_ok({"n": 7}))

    obs, _info = env.reset(seed=1)
    assert obs == {"n": 7}

    obs, reward, _term, _trunc, info = env.step({"type": "bump"})
    assert obs == {"n": 7}
    assert reward == 1.0
    assert info["status_code"] == 200


def test_every_container_call_carries_a_real_timeout():
    # The point of the routing. `RestTransport` maps an unset per-request
    # timeout to httpx's USE_CLIENT_DEFAULT sentinel, because passing `None`
    # means *no timeout* to httpx — a hung container would block forever.
    # Asserting on every sub-field so a partial timeout cannot pass.
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, json={"n": 0})

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=4.0)
    env = ContainerEnvBase("http://app", client=client)
    env.reset()
    env.step({"type": "bump"})

    assert seen, "no requests were made"
    for timeout in seen:
        assert timeout == {"connect": 4.0, "read": 4.0, "write": 4.0, "pool": 4.0}


# ---------------------------------------------------------------------------
# Preserved semantics: what raised before must still raise
# ---------------------------------------------------------------------------

def test_a_wire_failure_during_reset_raises():
    def handler(request):
        raise httpx.ConnectError("container is not up")

    with pytest.raises(ContainerTransportError, match="/forge/reset"):
        _env(handler).reset()


def test_a_failed_reset_status_raises():
    def handler(request):
        return httpx.Response(503, json={})

    with pytest.raises(ContainerTransportError, match="503"):
        _env(handler).reset()


def test_a_wire_failure_during_a_step_raises():
    def handler(request):
        if request.url.path.startswith("/forge/"):
            return httpx.Response(200, json={"n": 0})
        raise httpx.ConnectError("container died mid-episode")

    env = _env(handler)
    env.reset()
    with pytest.raises(ContainerTransportError):
        env.step({"type": "bump"})


def test_a_failed_state_fetch_after_an_action_raises():
    calls = {"n": 0}

    def handler(request):
        if request.url.path == "/forge/state":
            calls["n"] += 1
            # Succeed for reset, fail for the post-action fetch.
            if calls["n"] > 1:
                return httpx.Response(500, json={})
            return httpx.Response(200, json={"n": 0})
        return httpx.Response(200, json={})

    env = _env(handler)
    env.reset()
    with pytest.raises(ContainerTransportError, match="500"):
        env.step({"type": "bump"})


def test_a_non_json_state_body_raises():
    def handler(request):
        if request.url.path == "/forge/state":
            return httpx.Response(200, text="<html>502 Bad Gateway</html>")
        return httpx.Response(200, json={})

    with pytest.raises(ContainerTransportError):
        _env(handler).reset()


# ---------------------------------------------------------------------------
# Preserved semantics: what did NOT raise before must still not raise
# ---------------------------------------------------------------------------

def test_a_rejected_action_costs_reward_not_the_episode():
    # This is the important negative case. A non-2xx on the ACTION post has
    # never raised — an invalid action costs the step's reward, not the run.
    # Routing through a transport that reports errors in-band must not
    # quietly turn this into a raise.
    def handler(request):
        if request.url.path.startswith("/forge/"):
            return httpx.Response(200, json={"n": 0})
        return httpx.Response(422, json={"detail": "invalid"})

    env = _env(handler)
    env.reset()

    obs, reward, _term, _trunc, info = env.step({"type": "bump"})

    assert reward == 0.0
    assert info["status_code"] == 422
    assert obs == {"n": 0}


# ---------------------------------------------------------------------------
# The public hooks keep working
# ---------------------------------------------------------------------------

def test_compute_reward_receives_the_response_status():
    seen = {}

    class Scored(ContainerEnvBase):
        def compute_reward(self, response, obs) -> float:
            seen["status"] = response.status_code
            return 0.5

    client = httpx.Client(transport=httpx.MockTransport(_ok()))
    env = Scored("http://app", client=client)
    env.reset()

    _obs, reward, *_ = env.step({"type": "bump"})

    assert reward == 0.5
    assert seen["status"] == 200


def test_a_subclass_action_endpoint_override_is_still_honored():
    posted: list[str] = []

    def handler(request):
        if request.url.path.startswith("/forge/"):
            return httpx.Response(200, json={"n": 0})
        posted.append(request.url.path)
        return httpx.Response(200, json={})

    class Routed(ContainerEnvBase):
        def action_endpoint(self, action: dict) -> str:
            return f"/api/{action['type']}"

    client = httpx.Client(transport=httpx.MockTransport(handler))
    env = Routed("http://app", client=client)
    env.reset()
    env.step({"type": "bump"})

    assert posted == ["/api/bump"]


def test_the_backend_still_takes_a_typed_action():
    # The ExecutionBackend contract declares `Action`, not a dict; routing
    # through the transport must not regress that to the wire form.
    env = _env(_ok({"n": 3}))

    result = env.backend.execute(Action.from_dict({"type": "bump"}), {}, None)

    assert result.state == {"n": 3}
