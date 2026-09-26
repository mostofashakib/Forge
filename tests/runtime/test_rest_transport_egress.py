"""The agent picks the action endpoint, so it must never pick the host.

`RestTransport` joins the container's base URL with the request target. A
target the agent controls, like ``@example.com/x``, turns
``http://127.0.0.1:5000`` into a URL whose host is ``example.com``: a live
internet call from inside an episode. Only origin-relative paths may leave.
"""
from __future__ import annotations

import httpx
import pytest

from forge.contracts import TransportRequest
from forge.runtime.rest_transport import RestTransport


def _recording_transport() -> tuple[RestTransport, list[httpx.URL]]:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return RestTransport("http://127.0.0.1:5000", client=client), seen


@pytest.mark.parametrize("target", [
    "@example.com/steal",
    ".example.com/steal",
    "example.com/steal",
    "http://example.com/steal",
    "",
])
def test_a_target_that_is_not_a_path_never_reaches_the_wire(target):
    transport, seen = _recording_transport()

    response = transport.call(TransportRequest(method="POST", target=target))

    assert seen == []
    assert response.status == 0
    assert response.error is not None


@pytest.mark.parametrize("target", ["/send_email", "//example.com/x", "/users/a@b.com"])
def test_a_path_target_stays_on_the_container(target):
    # False-positive guard: legitimate paths, including ones that look like
    # hosts after the leading slash, still reach the container unchanged.
    transport, seen = _recording_transport()

    response = transport.call(TransportRequest(method="POST", target=target))

    assert response.error is None
    assert [(url.host, url.port) for url in seen] == [("127.0.0.1", 5000)]
