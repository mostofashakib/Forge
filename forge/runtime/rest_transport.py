# forge/runtime/rest_transport.py
"""HTTP transport for container-backed environments."""
from __future__ import annotations

import json

import httpx

from forge.contracts import Transport, TransportRequest, TransportResponse
from forge.contracts.transport import DEFAULT_TIMEOUT_S


class RestTransport(Transport):
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_S)

    def call(self, request: TransportRequest) -> TransportResponse:
        # The target is joined onto the base URL by string, and the agent picks
        # it. Anything but an origin-relative path can rewrite the host (e.g.
        # "@example.com/x" makes example.com the host), so it never leaves.
        if not request.target.startswith("/"):
            return TransportResponse(
                status=0,
                body={},
                error=f"target {request.target!r} is not a path on the environment",
            )
        # `None` is not "use the client default" to httpx — it means no
        # timeout at all. `USE_CLIENT_DEFAULT` is the actual sentinel for
        # that, so an unset `TransportRequest.timeout` must map to it rather
        # than pass through as `None`, or a genuinely hung container blocks
        # forever instead of costing one step.
        timeout = (
            httpx.USE_CLIENT_DEFAULT if request.timeout is None else request.timeout
        )
        try:
            response = self._client.request(
                request.method,
                f"{self._base_url}{request.target}",
                json=request.payload or None,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            # In-band, so one wire failure costs a step rather than the episode.
            return TransportResponse(status=0, body={}, error=str(exc))
        if not response.content:
            return TransportResponse(status=response.status_code, body={})
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            # A non-JSON body (an HTML 502 from a proxy in front of the
            # container is the realistic case) must not raise out of call()
            # either — same in-band contract as a wire failure, though the
            # status code is real and worth keeping rather than zeroing out.
            return TransportResponse(
                status=response.status_code, body={}, error=str(exc)
            )
        return TransportResponse(status=response.status_code, body=body)

    def close(self) -> None:
        self._client.close()
