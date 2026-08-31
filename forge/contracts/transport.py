"""How the model talks to the environment."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class TransportRequest(BaseModel):
    method: str = "POST"
    target: str = ""
    payload: dict = Field(default_factory=dict)
    timeout: float | None = None


class TransportResponse(BaseModel):
    """A transport result.

    Failures are reported in-band via `error` rather than raised, so a runner
    can record a failed step and continue instead of losing the whole episode
    to an exception from the wire.
    """

    status: int = 0
    body: dict = Field(default_factory=dict)
    error: str | None = None


class Transport(ABC):
    """The wire between the controller and the environment.

    Optional on the Environment facade: a pure-Python environment is reached by
    direct call and has no wire.
    """

    @abstractmethod
    def call(self, request: TransportRequest) -> TransportResponse:
        """Perform one round trip."""

    def close(self) -> None:
        """Release the connection. No-op by default."""
        return None
