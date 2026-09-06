"""What a verifier is allowed to look at.

An `Episode` is the graded record of one run: the world's end state and the
world's log of what the agent did. Both are produced by the environment, in a
container the agent has no filesystem or socket path to, which is what lets
trajectory and side-effect checks mean anything. An agent-authored trajectory
file would be evidence written by the party being graded.

There is no second input. An earlier version carried an optional agent-authored
artifact beside the state, which meant a verifier could ask "what did the agent
claim" rather than "what happened"; nothing needed it and it is gone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Action:
    """One tool call, as the world recorded it."""

    seq: int
    ts: str
    actor_id: str
    tool: str
    conversation_id: str | None = None
    thread_id: str | None = None
    message_id: str | None = None
    body_length: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Action":
        return cls(
            seq=int(row.get("seq", 0)),
            ts=str(row.get("ts", "")),
            actor_id=str(row.get("actor_id", "")),
            tool=str(row.get("tool", "")),
            conversation_id=row.get("conversation_id"),
            thread_id=row.get("thread_id"),
            message_id=row.get("message_id"),
            body_length=int(row.get("body_length") or 0),
        )


@dataclass(frozen=True, slots=True)
class Episode:
    state: dict[str, Any]
    actions: tuple[Action, ...] = ()
    actor_id: str = "U002"
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Whether the world recorded an action log at all. An export from before
    #: the log existed is a grading gap; an empty log is an agent that did
    #: nothing. Scoring both as zero would blame a model for a schema change.
    has_action_log: bool = True

    # -- construction ----------------------------------------------------

    @classmethod
    def from_state(cls, state: dict[str, Any], actor_id: str = "U002") -> "Episode":
        payload = state.get("result", state) if isinstance(state, dict) else {}
        actions = tuple(Action.from_row(row) for row in payload.get("action_log", []))
        return cls(
            state=payload, actions=actions, actor_id=actor_id,
            has_action_log="action_log" in payload,
        )

    @classmethod
    def from_files(cls, state_path: str | Path, actor_id: str = "U002") -> "Episode":
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        return cls.from_state(state, actor_id)

    # -- views the verifiers share ---------------------------------------

    @property
    def events(self) -> dict[str, str]:
        return {
            str(event["event_id"]): str(event["status"])
            for event in self.state.get("scenario_events", [])
        }

    @property
    def activated(self) -> set[str]:
        return {event for event, status in self.events.items() if status == "activated"}

    def activation_ts(self, event_id: str) -> str | None:
        for event in self.state.get("scenario_events", []):
            if event["event_id"] == event_id:
                return event.get("activated_ts")
        return None

    @property
    def messages(self) -> dict[str, dict[str, Any]]:
        return {str(m["message_id"]): m for m in self.state.get("messages", [])}

    def replies_in(self, thread_id: str, author_id: str | None = None) -> list[dict[str, Any]]:
        return [
            m for m in self.state.get("messages", [])
            if m.get("thread_parent_id") == thread_id
            and (author_id is None or m.get("author_id") == author_id)
        ]

    @property
    def integrity_violations(self) -> list[dict[str, Any]]:
        """Attempts to leave the environment, as the world recorded them."""
        return list(self.state.get("integrity_violations", []))

    @property
    def agent_actions(self) -> tuple[Action, ...]:
        return tuple(a for a in self.actions if a.actor_id == self.actor_id)

    def actions_with(self, tool: str) -> tuple[Action, ...]:
        return tuple(a for a in self.agent_actions if a.tool == tool)

    @property
    def tools_used(self) -> set[str]:
        return {a.tool for a in self.agent_actions}

    def channel(self, channel_id: str) -> dict[str, Any] | None:
        for channel in self.state.get("channels", []):
            if channel.get("channel_id") == channel_id:
                return channel
        return None

    def is_member(self, channel_id: str, user_id: str | None = None) -> bool:
        user_id = user_id or self.actor_id
        return any(
            m.get("channel_id") == channel_id and m.get("user_id") == user_id
            for m in self.state.get("memberships", [])
        )

    def namespace(self) -> dict[str, Any]:
        """The names a `PolicyVerifier` expression may use."""
        return {
            "state": self.state,
            "events": self.events,
            "activated": self.activated,
            "messages": self.messages,
            "actions": self.agent_actions,
            "tools_used": self.tools_used,
            "actor_id": self.actor_id,
            "episode": self,
            "len": len, "any": any, "all": all, "sorted": sorted, "sum": sum,
            "set": set, "bool": bool, "int": int, "float": float, "str": str,
        }
