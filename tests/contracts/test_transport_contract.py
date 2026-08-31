from __future__ import annotations

import pytest

from forge.contracts import Transport, TransportRequest, TransportResponse


class _Loopback(Transport):
    def call(self, request: TransportRequest) -> TransportResponse:
        return TransportResponse(status=200, body={"target": request.target})


def test_a_transport_round_trips_a_request():
    response = _Loopback().call(TransportRequest(method="POST", target="/close"))
    assert response.status == 200
    assert response.body == {"target": "/close"}
    assert response.error is None


def test_a_transport_response_can_carry_an_error():
    # Negative: transport failures are reported in-band, not by raising, so a
    # runner can record the failed step rather than losing the episode.
    response = TransportResponse(status=0, body={}, error="connection refused")
    assert response.error == "connection refused"


def test_a_transport_missing_call_cannot_be_instantiated():
    class Incomplete(Transport):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
