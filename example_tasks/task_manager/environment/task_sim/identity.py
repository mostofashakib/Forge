"""Identity configuration for the tracker user assigned to each rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrackerUserCredentials:
    user_id: str
    display_name: str
    email: str
    role: str
    team: str
    handle: str

    def as_row(self) -> tuple[Any, ...]:
        return (
            self.user_id, self.display_name, self.email,
            self.role, self.team, self.handle,
        )


#: Avery Chen, the workspace admin. The actor is decided by the socket a request
#: arrives on and is never read out of a tool payload, so naming an actor in one
#: is recorded as an integrity violation rather than honoured.
LOGGED_IN_USER = TrackerUserCredentials(
    user_id="U001",
    display_name="Avery Chen",
    email="avery@example.local",
    role="admin",
    team="research",
    handle="avery",
)
