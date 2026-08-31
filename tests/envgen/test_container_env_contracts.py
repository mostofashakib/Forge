# tests/envgen/test_container_env_contracts.py
from __future__ import annotations

import httpx
import pytest

from forge.contracts import (
    Action,
    Environment,
    StateManager,
    Transport,
    TransportRequest,
)
from forge.envgen.container_env_base import ContainerEnvBase


def _env() -> ContainerEnvBase:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tickets": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ContainerEnvBase("http://env", client=client)


def test_a_container_env_satisfies_the_environment_facade():
    assert isinstance(_env(), Environment)


def test_it_exposes_an_http_state_manager():
    assert isinstance(_env().state, StateManager)


def test_a_container_env_has_a_transport_because_it_is_over_a_wire():
    # False-positive guard: the optional members are optional in general, but
    # this family genuinely has one and must expose it.
    assert isinstance(_env().transport, Transport)


def test_reset_and_step_still_work_over_http():
    env = _env()
    obs, _info = env.reset()
    assert obs == {"tickets": []}
    obs, reward, terminated, truncated, info = env.step({"type": "close_ticket"})
    assert info["status_code"] == 200
    assert reward == 1.0


def test_a_wire_failure_is_reported_in_band_instead_of_raising():
    # Negative case: RestTransport must not let a connection failure blow up
    # the caller — a flaky container should cost one step, not the episode.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    env = ContainerEnvBase("http://env", client=client)

    response = env.transport.call(TransportRequest(method="GET", target="/forge/state"))

    assert response.error is not None
    assert response.status == 0


def test_an_http_error_status_is_not_mistaken_for_a_wire_failure():
    # False-positive guard: a container that responds (even with a 500) has
    # not had a wire failure, so `error` must stay unset and the real status
    # must come through.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    env = ContainerEnvBase("http://env", client=client)

    response = env.transport.call(TransportRequest(method="GET", target="/forge/state"))

    assert response.status == 500
    assert response.error is None


def test_the_initial_state_provider_actually_resets_over_http():
    # Pins the fix round 1 defect: `initial_state.reset` used to return `{}`
    # without touching the network at all. It must really POST /forge/reset
    # and hand back the state GET /forge/state returns afterward.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/forge/reset":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/forge/state":
            return httpx.Response(200, json={"tickets": ["t1"]})
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    env = ContainerEnvBase("http://env", client=client)

    state = env.initial_state.reset(None, seed=None, options={})

    assert calls == ["/forge/reset", "/forge/state"]
    assert state == {"tickets": ["t1"]}


def test_the_backend_actually_executes_the_action_over_http():
    # Pins the fix round 1 defect: `backend.execute` used to echo the input
    # state back unchanged without touching the network at all. It must
    # really POST the action and hand back the state GET /forge/state
    # returns afterward. Called with a real `Action`, the way the
    # `ExecutionBackend` contract declares — not the dict `step()` happens to
    # hand it internally.
    calls: list[str] = []
    state = {"tickets": []}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/close_ticket":
            state["tickets"] = ["closed"]
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/forge/state":
            return httpx.Response(200, json=state)
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    env = ContainerEnvBase("http://env", client=client)

    result = env.backend.execute(Action(type="close_ticket"), {}, None)

    assert calls == ["/close_ticket", "/forge/state"]
    assert result.state == {"tickets": ["closed"]}


def test_the_backend_rejects_a_plain_dict_because_it_takes_a_typed_action():
    # Pins the fix round 2 defect directly: calling `execute` through its own
    # declared contract (`action: Action`) must work; a plain dict is not
    # that contract, even though `step()` happens to hold one internally.
    env = _env()

    with pytest.raises(AttributeError):
        env.backend.execute({"type": "close_ticket"}, {}, None)


def test_a_subclass_action_endpoint_override_is_honored_by_the_backend():
    # False-positive guard: the backend must route through the env's own
    # action_endpoint hook rather than a hardcoded default, or a subclass's
    # routing customization would silently stop applying once behind the
    # facade. `action_endpoint` itself still receives a plain dict — only the
    # backend's own `execute` boundary takes the typed `Action`.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/create_todo":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/forge/state":
            return httpx.Response(200, json={"todos": {}})
        return httpx.Response(404, json={})

    class TodoEnv(ContainerEnvBase):
        def action_endpoint(self, action: dict) -> str:
            return "/create_todo"  # domain routes everything to one endpoint

    client = httpx.Client(transport=httpx.MockTransport(handler))
    env = TodoEnv("http://env", client=client)

    env.backend.execute(Action(type="anything"), {}, None)

    assert calls[0] == "/create_todo"


def test_step_still_works_with_a_plain_dict_action():
    # False-positive guard: converting to Action at the backend boundary must
    # not break step()'s public, dict-based entry point.
    env = _env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(
        {"type": "close_ticket", "note": "done"}
    )
    assert info["status_code"] == 200
    assert reward == 1.0


def test_an_unset_timeout_uses_the_clients_configured_timeout_not_none():
    # Pins the fix round 3 defect: httpx treats an explicit `timeout=None` as
    # "no timeout at all," not "use the client default." A TransportRequest
    # with no timeout must map to httpx.USE_CLIENT_DEFAULT so a genuinely
    # hung container still times out instead of blocking forever.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=15.0)
    env = ContainerEnvBase("http://env", client=client)

    env.transport.call(TransportRequest(method="GET", target="/forge/state"))

    assert seen["timeout"] == {
        "connect": 15.0,
        "read": 15.0,
        "write": 15.0,
        "pool": 15.0,
    }


def test_an_explicit_timeout_still_overrides_the_client_default():
    # False-positive guard: fixing the unset case must not stop an explicit,
    # shorter per-call timeout from being honored.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=15.0)
    env = ContainerEnvBase("http://env", client=client)

    env.transport.call(
        TransportRequest(method="GET", target="/forge/state", timeout=2.0)
    )

    assert seen["timeout"] == {"connect": 2.0, "read": 2.0, "write": 2.0, "pool": 2.0}


def test_a_non_json_error_body_is_reported_in_band_not_raised():
    # Pins the fix round 3 defect: a non-JSON body (an HTML 502 from a proxy
    # in front of the container is the realistic case) must not raise out of
    # call() either — same in-band contract as a wire failure — and the real
    # status code must still come through.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502, content=b"<html>Bad Gateway</html>", headers={"content-type": "text/html"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    env = ContainerEnvBase("http://env", client=client)

    response = env.transport.call(TransportRequest(method="GET", target="/forge/state"))

    assert response.status == 502
    assert response.error is not None
    assert response.body == {}


def test_http_action_result_serializes_despite_carrying_a_response():
    # Pins the fix round 3 defect: _HttpActionResult used to raise
    # PydanticSerializationError on model_dump(mode="json") /
    # model_dump_json() because httpx.Response isn't JSON-serializable.
    # Nothing serializes an ActionResult today, but the first generic
    # consumer that does (a trajectory logger, a replay dump) must not crash
    # only for the container family.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/close_ticket":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json={"tickets": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    env = ContainerEnvBase("http://env", client=client)

    result = env.backend.execute(Action(type="close_ticket"), {}, None)

    assert result.model_dump() == {"state": {"tickets": []}, "events": [], "error": None}
    assert result.model_dump(mode="json") == {
        "state": {"tickets": []},
        "events": [],
        "error": None,
    }
    assert result.model_dump_json() is not None


def test_http_action_result_response_is_still_readable_after_serializing():
    # False-positive guard: excluding `response` from serialization must not
    # mean dropping the attribute itself — step() still needs it afterward.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tickets": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    env = ContainerEnvBase("http://env", client=client)

    result = env.backend.execute(Action(type="close_ticket"), {}, None)
    result.model_dump_json()  # serialize first

    assert result.response.status_code == 200
