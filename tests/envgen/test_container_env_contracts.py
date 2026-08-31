# tests/envgen/test_container_env_contracts.py
from __future__ import annotations

import httpx

from forge.contracts import Environment, StateManager, Transport, TransportRequest
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
