# forge/runtime/rest_transport.py
"""HTTP transport for container-backed environments."""
from __future__ import annotations

import httpx

from forge.contracts import Transport, TransportRequest, TransportResponse


class RestTransport(Transport):
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=15.0)

    def call(self, request: TransportRequest) -> TransportResponse:
        try:
            response = self._client.request(
                request.method,
                f"{self._base_url}{request.target}",
                json=request.payload or None,
                timeout=request.timeout,
            )
        except httpx.HTTPError as exc:
            # In-band, so one wire failure costs a step rather than the episode.
            return TransportResponse(status=0, body={}, error=str(exc))
        body = response.json() if response.content else {}
        return TransportResponse(status=response.status_code, body=body)

    def close(self) -> None:
        self._client.close()
