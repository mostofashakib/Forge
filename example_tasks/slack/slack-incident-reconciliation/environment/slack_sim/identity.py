"""Identity configuration for the Slack user assigned to each rollout."""

from __future__ import annotations

from dataclasses import dataclass

from slack_sim.models import SlackUser


@dataclass(frozen=True, slots=True)
class SlackUserCredentials:
    user_id: str
    display_name: str
    email: str
    role: str
    team: str
    handle: str

    def as_user(self) -> SlackUser:
        return SlackUser(
            user_id=self.user_id,
            display_name=self.display_name,
            email=self.email,
            role=self.role,
            team=self.team,
            handle=self.handle,
        )


LOGGED_IN_USER = SlackUserCredentials(
    user_id="U002",
    display_name="Ben Ortiz",
    email="ben@example.local",
    role="member",
    team="platform",
    handle="ben",
)
