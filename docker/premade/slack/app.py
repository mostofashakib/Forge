"""Slack-like Forge RL environment.

Simulates a team messaging workspace with channels, direct messages, threads,
reactions, pins, and user status. Fully compatible with the Forge
ContainerEpisodeRunner protocol.

Forge endpoints:
  GET  /forge/health  → 200 OK
  GET  /forge/state   → current workspace state
  POST /forge/reset   → reset to seed state

Action endpoints (all POST, all return updated state):
  /send_message   /reply_thread  /add_reaction    /remove_reaction
  /archive_channel /create_channel /pin_message   /delete_message
  /set_status     /mark_channel_read
"""
from __future__ import annotations
import copy
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Slack-like Environment", description="Forge RL environment simulating a team messaging workspace.")

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_CHANNELS: list[dict] = [
    {
        "id": "C001", "name": "general", "purpose": "Company-wide announcements and watercooler chat",
        "archived": False, "unread": 3, "pinned_messages": [],
        "messages": [
            {"id": "m001", "user": "alice", "text": "Good morning everyone! Reminder: all-hands at 3pm today.", "timestamp": "2026-05-01T08:00:00Z", "reactions": {"👋": ["bob", "carol"]}, "thread_replies": 0, "pinned": False},
            {"id": "m002", "user": "bob", "text": "Thanks for the reminder Alice! Will the recording be shared?", "timestamp": "2026-05-01T08:05:00Z", "reactions": {}, "thread_replies": 1, "pinned": False},
            {"id": "m003", "user": "carol", "text": "The Q3 results are looking great! We hit 120% of our target 🎉", "timestamp": "2026-05-01T09:30:00Z", "reactions": {"🎉": ["alice", "dave", "eve"], "👏": ["alice"]}, "thread_replies": 0, "pinned": False},
            {"id": "m004", "user": "dave", "text": "Quick heads up — prod deploy is happening at 5pm. Expect ~10min downtime.", "timestamp": "2026-05-01T14:00:00Z", "reactions": {"👍": ["alice"]}, "thread_replies": 2, "pinned": False},
            {"id": "m005", "user": "me", "text": "I'll be OOO Friday. Pinging @alice to cover.", "timestamp": "2026-05-01T15:30:00Z", "reactions": {}, "thread_replies": 0, "pinned": False},
        ],
    },
    {
        "id": "C002", "name": "engineering", "purpose": "Technical discussions and code reviews",
        "archived": False, "unread": 1, "pinned_messages": [],
        "messages": [
            {"id": "m010", "user": "eve", "text": "PR #247 is ready for review: https://github.com/company/forge/pull/247", "timestamp": "2026-05-01T10:00:00Z", "reactions": {"👀": ["alice"]}, "thread_replies": 3, "pinned": False},
            {"id": "m011", "user": "alice", "text": "Heads up: we're upgrading to Python 3.12 next sprint.", "timestamp": "2026-05-01T11:00:00Z", "reactions": {}, "thread_replies": 0, "pinned": False},
            {"id": "m012", "user": "bob", "text": "The flaky test in CI has been fixed. Root cause was a race condition in the auth module.", "timestamp": "2026-05-01T16:00:00Z", "reactions": {"🙌": ["carol", "dave"]}, "thread_replies": 0, "pinned": False},
        ],
    },
    {
        "id": "C003", "name": "announcements", "purpose": "Important company announcements — admins only",
        "archived": False, "unread": 0, "pinned_messages": ["m020"],
        "messages": [
            {"id": "m020", "user": "ceo", "text": "Welcome to our new VP of Engineering, Sarah Chen! She joins us from Google with 15 years of experience.", "timestamp": "2026-04-28T09:00:00Z", "reactions": {"🎉": ["alice", "bob", "carol", "dave", "eve"]}, "thread_replies": 0, "pinned": True},
            {"id": "m021", "user": "hr", "text": "Reminder: performance reviews are due by May 31st. Please complete them in Lattice.", "timestamp": "2026-04-30T10:00:00Z", "reactions": {"✅": ["alice", "bob"]}, "thread_replies": 0, "pinned": False},
        ],
    },
    {
        "id": "C004", "name": "random", "purpose": "Non-work banter and fun",
        "archived": False, "unread": 5, "pinned_messages": [],
        "messages": [
            {"id": "m030", "user": "carol", "text": "Anyone watching the game tonight? 🏀", "timestamp": "2026-05-01T12:00:00Z", "reactions": {"🏀": ["bob", "dave"]}, "thread_replies": 4, "pinned": False},
            {"id": "m031", "user": "dave", "text": "Hot take: tabs are better than spaces", "timestamp": "2026-05-01T13:00:00Z", "reactions": {"😂": ["alice", "carol"], "🔥": ["eve"]}, "thread_replies": 7, "pinned": False},
        ],
    },
]

_SEED_DMS: list[dict] = [
    {"id": "dm_alice", "with": "alice", "unread": 2, "messages": [
        {"id": "dm001", "from": "alice", "text": "Hey, do you have a minute to chat about the Q3 presentation?", "timestamp": "2026-05-01T11:30:00Z"},
        {"id": "dm002", "from": "alice", "text": "It's about the slide deck for Friday.", "timestamp": "2026-05-01T11:31:00Z"},
    ]},
    {"id": "dm_bob", "with": "bob", "unread": 0, "messages": [
        {"id": "dm010", "from": "me", "text": "Thanks for the code review!", "timestamp": "2026-04-30T16:00:00Z"},
        {"id": "dm011", "from": "bob", "text": "Happy to help! Great work on the refactor.", "timestamp": "2026-04-30T16:05:00Z"},
    ]},
]

_SEED_USER = {"name": "me", "status": "", "status_emoji": "", "presence": "online"}

_state: dict = {}


def _reset_state() -> None:
    global _state
    _state = {
        "channels": copy.deepcopy(_SEED_CHANNELS),
        "direct_messages": copy.deepcopy(_SEED_DMS),
        "user": dict(_SEED_USER),
    }


_reset_state()


def _channel_by_id(channel_id: str) -> dict | None:
    return next((c for c in _state["channels"] if c["id"] == channel_id or c["name"] == channel_id), None)


def _msg_in_channel(channel: dict, msg_id: str) -> dict | None:
    return next((m for m in channel["messages"] if m["id"] == msg_id), None)


def _snapshot() -> dict:
    total_unread = sum(c["unread"] for c in _state["channels"]) + sum(d["unread"] for d in _state["direct_messages"])
    return {
        "channels": [
            {"id": c["id"], "name": c["name"], "archived": c["archived"],
             "unread": c["unread"], "message_count": len(c["messages"]),
             "pinned_count": len(c["pinned_messages"])}
            for c in _state["channels"]
        ],
        "total_unread": total_unread,
        "dm_unread": sum(d["unread"] for d in _state["direct_messages"]),
        "user": _state["user"],
    }


# ---------------------------------------------------------------------------
# Forge protocol
# ---------------------------------------------------------------------------

@app.get("/forge/health")
def health():
    return {"status": "ok"}


@app.get("/forge/state")
def get_state():
    return _snapshot()


@app.post("/forge/reset")
def reset():
    _reset_state()
    return {"status": "reset", "state": _snapshot()}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class SendMessageRequest(BaseModel):
    channel: str
    text: str


@app.post("/send_message", summary="Send a message to a channel")
def send_message(req: SendMessageRequest):
    channel = _channel_by_id(req.channel)
    if not channel:
        return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _snapshot()}
    if channel["archived"]:
        return {"status": "error", "message": f"Channel '{req.channel}' is archived", "state": _snapshot()}
    msg_id = f"m{uuid.uuid4().hex[:6]}"
    channel["messages"].append({
        "id": msg_id, "user": "me", "text": req.text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reactions": {}, "thread_replies": 0, "pinned": False,
    })
    return {"status": "sent", "message_id": msg_id, "channel": req.channel, "state": _snapshot()}


class ReplyThreadRequest(BaseModel):
    channel: str
    message_id: str
    text: str


@app.post("/reply_thread", summary="Reply to a message in a thread")
def reply_thread(req: ReplyThreadRequest):
    channel = _channel_by_id(req.channel)
    if not channel:
        return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _snapshot()}
    original = _msg_in_channel(channel, req.message_id)
    if not original:
        return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _snapshot()}
    original["thread_replies"] += 1
    reply_id = f"t{uuid.uuid4().hex[:6]}"
    return {"status": "replied", "reply_id": reply_id, "thread_of": req.message_id, "state": _snapshot()}


class ReactionRequest(BaseModel):
    channel: str
    message_id: str
    emoji: str


@app.post("/add_reaction", summary="Add an emoji reaction to a message")
def add_reaction(req: ReactionRequest):
    channel = _channel_by_id(req.channel)
    if not channel:
        return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _snapshot()}
    msg = _msg_in_channel(channel, req.message_id)
    if not msg:
        return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _snapshot()}
    if req.emoji not in msg["reactions"]:
        msg["reactions"][req.emoji] = []
    if "me" not in msg["reactions"][req.emoji]:
        msg["reactions"][req.emoji].append("me")
    return {"status": "reacted", "emoji": req.emoji, "message_id": req.message_id, "state": _snapshot()}


@app.post("/remove_reaction", summary="Remove an emoji reaction from a message")
def remove_reaction(req: ReactionRequest):
    channel = _channel_by_id(req.channel)
    if not channel:
        return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _snapshot()}
    msg = _msg_in_channel(channel, req.message_id)
    if not msg:
        return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _snapshot()}
    if req.emoji in msg["reactions"] and "me" in msg["reactions"][req.emoji]:
        msg["reactions"][req.emoji].remove("me")
    return {"status": "removed", "emoji": req.emoji, "message_id": req.message_id, "state": _snapshot()}


class ChannelRequest(BaseModel):
    channel: str


@app.post("/archive_channel", summary="Archive a channel")
def archive_channel(req: ChannelRequest):
    channel = _channel_by_id(req.channel)
    if not channel:
        return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _snapshot()}
    channel["archived"] = True
    return {"status": "archived", "channel": req.channel, "state": _snapshot()}


class CreateChannelRequest(BaseModel):
    name: str
    purpose: str = ""


@app.post("/create_channel", summary="Create a new channel")
def create_channel(req: CreateChannelRequest):
    existing = _channel_by_id(req.name)
    if existing:
        return {"status": "error", "message": f"Channel '{req.name}' already exists", "state": _snapshot()}
    channel_id = f"C{uuid.uuid4().hex[:3].upper()}"
    _state["channels"].append({
        "id": channel_id, "name": req.name.lower().replace(" ", "-"),
        "purpose": req.purpose, "archived": False, "unread": 0,
        "pinned_messages": [], "messages": [],
    })
    return {"status": "created", "channel_id": channel_id, "name": req.name, "state": _snapshot()}


class PinRequest(BaseModel):
    channel: str
    message_id: str


@app.post("/pin_message", summary="Pin a message in a channel")
def pin_message(req: PinRequest):
    channel = _channel_by_id(req.channel)
    if not channel:
        return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _snapshot()}
    msg = _msg_in_channel(channel, req.message_id)
    if not msg:
        return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _snapshot()}
    msg["pinned"] = True
    if req.message_id not in channel["pinned_messages"]:
        channel["pinned_messages"].append(req.message_id)
    return {"status": "pinned", "message_id": req.message_id, "channel": req.channel, "state": _snapshot()}


class DeleteMessageRequest(BaseModel):
    channel: str
    message_id: str


@app.post("/delete_message", summary="Delete a message from a channel")
def delete_message(req: DeleteMessageRequest):
    channel = _channel_by_id(req.channel)
    if not channel:
        return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _snapshot()}
    before = len(channel["messages"])
    channel["messages"] = [m for m in channel["messages"] if m["id"] != req.message_id]
    if len(channel["messages"]) == before:
        return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _snapshot()}
    if req.message_id in channel["pinned_messages"]:
        channel["pinned_messages"].remove(req.message_id)
    return {"status": "deleted", "message_id": req.message_id, "state": _snapshot()}


class SetStatusRequest(BaseModel):
    status: str
    emoji: str = ""


@app.post("/set_status", summary="Set your user status message")
def set_status(req: SetStatusRequest):
    _state["user"]["status"] = req.status
    _state["user"]["status_emoji"] = req.emoji
    return {"status": "updated", "new_status": req.status, "emoji": req.emoji, "state": _snapshot()}


@app.post("/mark_channel_read", summary="Mark all messages in a channel as read")
def mark_channel_read(req: ChannelRequest):
    channel = _channel_by_id(req.channel)
    if not channel:
        return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _snapshot()}
    channel["unread"] = 0
    return {"status": "marked_read", "channel": req.channel, "state": _snapshot()}
