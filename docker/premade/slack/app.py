"""Slack-like Forge RL environment — high-fidelity SQLAlchemy/SQLite version.

Forge protocol:
  GET  /forge/health
  GET  /forge/state
  POST /forge/reset
  POST /forge/snapshot        body: {"slot": "name"}
  POST /forge/restore/{slot}
  POST /forge/restore-state   body: full state JSON

Action endpoints (all POST):
  /send_message  /reply_thread  /add_reaction  /remove_reaction
  /pin_message   /unpin_message  /delete_message  /create_channel
  /archive_channel  /set_status  /send_dm  /mark_channel_read
  /search_messages  /get_channel_messages (also GET)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import (
    Boolean, Column, Integer, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite:///./slack.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Channel(Base):
    __tablename__ = "channels"

    id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    purpose = Column(String, default="")
    is_private = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)
    created_at = Column(String, nullable=False)


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    channel_id = Column(String, nullable=False)
    user_name = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    timestamp = Column(String, nullable=False)
    is_pinned = Column(Boolean, default=False)
    thread_parent_id = Column(String, nullable=True)  # None = top-level
    reply_count = Column(Integer, default=0)


class Reaction(Base):
    __tablename__ = "reactions"

    id = Column(String, primary_key=True, default=lambda: f"rx{uuid.uuid4().hex[:8]}")
    message_id = Column(String, nullable=False)
    emoji = Column(String, nullable=False)
    user_name = Column(String, nullable=False)


class DirectMessage(Base):
    __tablename__ = "direct_messages"

    id = Column(String, primary_key=True)
    from_user = Column(String, nullable=False)
    to_user = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    timestamp = Column(String, nullable=False)
    is_read = Column(Boolean, default=False)


class UserStatus(Base):
    __tablename__ = "user_status"

    user_name = Column(String, primary_key=True)
    status_text = Column(String, default="")
    status_emoji = Column(String, default="")
    presence = Column(String, default="online")


class ChannelReadState(Base):
    """Tracks per-channel unread count."""
    __tablename__ = "channel_read_state"

    channel_id = Column(String, primary_key=True)
    unread_count = Column(Integer, default=0)


class SavedState(Base):
    __tablename__ = "saved_states"

    slot = Column(String, primary_key=True)
    data = Column(Text, nullable=False)


Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Slack-like Environment", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _channel_to_dict(c: Channel, db: Session) -> dict:
    msg_count = db.query(Message).filter(
        Message.channel_id == c.id,
        Message.thread_parent_id == None  # noqa: E711
    ).count()
    pinned_count = db.query(Message).filter(
        Message.channel_id == c.id,
        Message.is_pinned == True  # noqa: E712
    ).count()
    rs = db.query(ChannelReadState).filter(ChannelReadState.channel_id == c.id).first()
    unread = rs.unread_count if rs else 0
    return {
        "id": c.id,
        "name": c.name,
        "purpose": c.purpose,
        "is_private": c.is_private,
        "archived": c.is_archived,
        "message_count": msg_count,
        "pinned_count": pinned_count,
        "unread": unread,
    }


def _message_to_dict(m: Message, db: Session) -> dict:
    reactions_raw = db.query(Reaction).filter(Reaction.message_id == m.id).all()
    reactions: dict[str, list[str]] = {}
    for r in reactions_raw:
        reactions.setdefault(r.emoji, []).append(r.user_name)
    return {
        "id": m.id,
        "channel_id": m.channel_id,
        "user": m.user_name,
        "text": m.text,
        "timestamp": m.timestamp,
        "is_pinned": m.is_pinned,
        "thread_parent_id": m.thread_parent_id,
        "reply_count": m.reply_count,
        "reactions": reactions,
    }


def _dm_to_dict(dm: DirectMessage) -> dict:
    return {
        "id": dm.id,
        "from": dm.from_user,
        "to": dm.to_user,
        "text": dm.text,
        "timestamp": dm.timestamp,
        "is_read": dm.is_read,
    }


def _get_state_dict(db: Session) -> dict:
    channels = db.query(Channel).all()
    status = db.query(UserStatus).filter(UserStatus.user_name == "me").first()
    user_status = {
        "user_name": "me",
        "status_text": status.status_text if status else "",
        "status_emoji": status.status_emoji if status else "",
        "presence": status.presence if status else "online",
    }
    dm_unread = db.query(DirectMessage).filter(
        DirectMessage.to_user == "me",
        DirectMessage.is_read == False  # noqa: E712
    ).count()
    total_unread = dm_unread + sum(
        (db.query(ChannelReadState).filter(ChannelReadState.channel_id == c.id).first() or ChannelReadState(unread_count=0)).unread_count
        for c in channels
    )
    return {
        "workspace_name": "Acme Corp",
        "channels": [_channel_to_dict(c, db) for c in channels],
        "total_unread": total_unread,
        "dm_unread": dm_unread,
        "user_status": user_status,
    }


def _dump_full_db(db: Session) -> dict:
    channels = [
        {"id": c.id, "name": c.name, "purpose": c.purpose,
         "is_private": c.is_private, "is_archived": c.is_archived, "created_at": c.created_at}
        for c in db.query(Channel).all()
    ]
    messages = [
        {"id": m.id, "channel_id": m.channel_id, "user_name": m.user_name,
         "text": m.text, "timestamp": m.timestamp, "is_pinned": m.is_pinned,
         "thread_parent_id": m.thread_parent_id, "reply_count": m.reply_count}
        for m in db.query(Message).all()
    ]
    reactions = [
        {"id": r.id, "message_id": r.message_id, "emoji": r.emoji, "user_name": r.user_name}
        for r in db.query(Reaction).all()
    ]
    dms = [
        {"id": dm.id, "from_user": dm.from_user, "to_user": dm.to_user,
         "text": dm.text, "timestamp": dm.timestamp, "is_read": dm.is_read}
        for dm in db.query(DirectMessage).all()
    ]
    statuses = [
        {"user_name": s.user_name, "status_text": s.status_text,
         "status_emoji": s.status_emoji, "presence": s.presence}
        for s in db.query(UserStatus).all()
    ]
    read_states = [
        {"channel_id": rs.channel_id, "unread_count": rs.unread_count}
        for rs in db.query(ChannelReadState).all()
    ]
    return {
        "channels": channels, "messages": messages, "reactions": reactions,
        "direct_messages": dms, "user_statuses": statuses, "read_states": read_states,
    }


def _restore_from_dict(db: Session, data: dict) -> None:
    db.query(Reaction).delete()
    db.query(Message).delete()
    db.query(Channel).delete()
    db.query(DirectMessage).delete()
    db.query(UserStatus).delete()
    db.query(ChannelReadState).delete()
    for c in data.get("channels", []):
        db.add(Channel(
            id=c["id"], name=c["name"], purpose=c.get("purpose", ""),
            is_private=c.get("is_private", False), is_archived=c.get("is_archived", c.get("archived", False)),
            created_at=c.get("created_at", _now()),
        ))
    for m in data.get("messages", []):
        db.add(Message(
            id=m["id"], channel_id=m["channel_id"], user_name=m.get("user_name", m.get("user", "unknown")),
            text=m["text"], timestamp=m["timestamp"], is_pinned=m.get("is_pinned", m.get("pinned", False)),
            thread_parent_id=m.get("thread_parent_id"), reply_count=m.get("reply_count", 0),
        ))
    for r in data.get("reactions", []):
        db.add(Reaction(id=r["id"], message_id=r["message_id"], emoji=r["emoji"], user_name=r["user_name"]))
    for dm in data.get("direct_messages", []):
        db.add(DirectMessage(
            id=dm["id"], from_user=dm.get("from_user", dm.get("from", "")),
            to_user=dm.get("to_user", dm.get("to", "")),
            text=dm["text"], timestamp=dm["timestamp"], is_read=dm.get("is_read", False),
        ))
    for s in data.get("user_statuses", []):
        db.add(UserStatus(
            user_name=s["user_name"], status_text=s.get("status_text", ""),
            status_emoji=s.get("status_emoji", ""), presence=s.get("presence", "online"),
        ))
    for rs in data.get("read_states", []):
        db.add(ChannelReadState(channel_id=rs["channel_id"], unread_count=rs.get("unread_count", 0)))
    db.commit()


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_CHANNELS = [
    {"id": "C001", "name": "general", "purpose": "Company-wide announcements and watercooler chat", "is_private": False, "is_archived": False},
    {"id": "C002", "name": "engineering", "purpose": "Technical discussions, code reviews, and deployments", "is_private": False, "is_archived": False},
    {"id": "C003", "name": "product", "purpose": "Product roadmap, feature discussions, and UX feedback", "is_private": False, "is_archived": False},
    {"id": "C004", "name": "random", "purpose": "Non-work banter and fun stuff", "is_private": False, "is_archived": False},
    {"id": "C005", "name": "announcements", "purpose": "Important company announcements — admins only", "is_private": False, "is_archived": False},
]

_SEED_MESSAGES = [
    # ---- #general ----
    {"id": "m001", "channel_id": "C001", "user_name": "alice", "text": "Good morning team! Reminder: all-hands at 3pm today. Zoom link in your calendar invite.", "timestamp": "2026-05-04T08:00:00Z", "is_pinned": True, "reply_count": 2},
    {"id": "m002", "channel_id": "C001", "user_name": "bob", "text": "Thanks Alice! Will the recording be shared afterward for folks in different timezones?", "timestamp": "2026-05-04T08:05:00Z", "is_pinned": False, "reply_count": 1},
    {"id": "m003", "channel_id": "C001", "user_name": "carol", "text": "Q2 results are in — we hit 118% of our revenue target! Amazing work everyone 🎉", "timestamp": "2026-05-04T09:30:00Z", "is_pinned": False, "reply_count": 0},
    {"id": "m004", "channel_id": "C001", "user_name": "dave", "text": "Heads up: I'm deploying the new dashboard at 5pm EST. Expect ~5 minute downtime on the metrics page.", "timestamp": "2026-05-04T14:00:00Z", "is_pinned": False, "reply_count": 3},
    {"id": "m005", "channel_id": "C001", "user_name": "me", "text": "I'll be OOO this Friday. Pinging @alice to cover any urgent issues on my side.", "timestamp": "2026-05-04T15:30:00Z", "is_pinned": False, "reply_count": 0},
    {"id": "m006", "channel_id": "C001", "user_name": "eve", "text": "The new onboarding docs are live! Check them out at docs.acme.com/onboarding — took the whole week but worth it.", "timestamp": "2026-05-03T16:45:00Z", "is_pinned": False, "reply_count": 2},
    {"id": "m007", "channel_id": "C001", "user_name": "alice", "text": "Big news: we just signed a contract with Northstar Capital. This is our largest enterprise deal to date!", "timestamp": "2026-05-02T11:00:00Z", "is_pinned": True, "reply_count": 5},

    # ---- #engineering ----
    {"id": "m010", "channel_id": "C002", "user_name": "eve", "text": "PR #312 is ready for review: https://github.com/acme/forge/pull/312 — adds SQLAlchemy persistence layer. ~400 LOC, all tests passing.", "timestamp": "2026-05-04T10:00:00Z", "is_pinned": False, "reply_count": 4},
    {"id": "m011", "channel_id": "C002", "user_name": "alice", "text": "Heads up: we're upgrading to Python 3.12 in the next sprint. Please audit any deps that might break.", "timestamp": "2026-05-04T11:00:00Z", "is_pinned": True, "reply_count": 2},
    {"id": "m012", "channel_id": "C002", "user_name": "bob", "text": "The flaky test in CI has been fixed! Root cause: race condition in the auth token refresh. Added a mutex. Took me 3 hours but finally cracked it 🎯", "timestamp": "2026-05-04T16:00:00Z", "is_pinned": False, "reply_count": 0},
    {"id": "m013", "channel_id": "C002", "user_name": "carol", "text": "Incident post-mortem for the DB spike is ready for review: https://docs.acme.com/postmortem-042 — please add comments by EOD tomorrow.", "timestamp": "2026-05-03T15:00:00Z", "is_pinned": False, "reply_count": 1},
    {"id": "m014", "channel_id": "C002", "user_name": "me", "text": "Just merged the async worker refactor. Should cut background job latency by 60%. Monitoring it now.", "timestamp": "2026-05-03T17:30:00Z", "is_pinned": False, "reply_count": 3},
    {"id": "m015", "channel_id": "C002", "user_name": "dave", "text": "Reminder: code freeze for v2.4.0 is this Thursday at noon. No more feature PRs after that — hotfixes only.", "timestamp": "2026-05-02T09:00:00Z", "is_pinned": True, "reply_count": 0},
    {"id": "m016", "channel_id": "C002", "user_name": "bob", "text": "Anyone familiar with clickhouse? We're evaluating it to replace our analytics Postgres. Would love a second opinion.", "timestamp": "2026-05-01T14:00:00Z", "is_pinned": False, "reply_count": 5},

    # ---- #product ----
    {"id": "m020", "channel_id": "C003", "user_name": "carol", "text": "Updated the Q3 roadmap in Notion. Key additions: AI-powered search, bulk actions for agents, and a new onboarding flow. Lmk thoughts!", "timestamp": "2026-05-04T10:30:00Z", "is_pinned": False, "reply_count": 6},
    {"id": "m021", "channel_id": "C003", "user_name": "dave", "text": "Mocked up 3 options for the new dashboard. Sharing Figma: https://figma.com/file/acme/dashboard-v2 — feedback appreciated!", "timestamp": "2026-05-04T13:00:00Z", "is_pinned": False, "reply_count": 4},
    {"id": "m022", "channel_id": "C003", "user_name": "me", "text": "User research call with 5 enterprise customers last week. Main pain points: slow episode resets and lack of bulk actions. Summarizing in a doc.", "timestamp": "2026-05-03T11:00:00Z", "is_pinned": False, "reply_count": 2},
    {"id": "m023", "channel_id": "C003", "user_name": "alice", "text": "NPS score this month: 72 (up from 61!). Biggest driver was the new snapshot/restore feature. Great job team!", "timestamp": "2026-05-02T16:00:00Z", "is_pinned": True, "reply_count": 3},
    {"id": "m024", "channel_id": "C003", "user_name": "carol", "text": "Can someone help me understand the agent observation format? Trying to document it for the API reference.", "timestamp": "2026-05-01T13:45:00Z", "is_pinned": False, "reply_count": 1},

    # ---- #random ----
    {"id": "m030", "channel_id": "C004", "user_name": "bob", "text": "Hot take: tabs are better than spaces and I will die on this hill", "timestamp": "2026-05-04T12:00:00Z", "is_pinned": False, "reply_count": 7},
    {"id": "m031", "channel_id": "C004", "user_name": "carol", "text": "Anyone watching the playoffs tonight? 🏀 We should do a watch party in the office!", "timestamp": "2026-05-04T13:00:00Z", "is_pinned": False, "reply_count": 4},
    {"id": "m032", "channel_id": "C004", "user_name": "dave", "text": "Just discovered you can ask Claude to play chess in the terminal and it actually works??? What is this timeline", "timestamp": "2026-05-04T14:30:00Z", "is_pinned": False, "reply_count": 3},
    {"id": "m033", "channel_id": "C004", "user_name": "eve", "text": "Friday lunch at the taco place? I'm buying if we hit the sprint goal 🌮", "timestamp": "2026-05-03T17:00:00Z", "is_pinned": False, "reply_count": 5},
    {"id": "m034", "channel_id": "C004", "user_name": "me", "text": "We hit the sprint goal! Tacos on Eve this Friday 🌮🌮🌮", "timestamp": "2026-05-03T18:00:00Z", "is_pinned": False, "reply_count": 0},

    # ---- #announcements ----
    {"id": "m040", "channel_id": "C005", "user_name": "alice", "text": "Welcome to our new VP of Engineering, Sarah Chen! She joins us from Google with 15 years of infrastructure experience. Sarah starts May 12th.", "timestamp": "2026-04-28T09:00:00Z", "is_pinned": True, "reply_count": 0},
    {"id": "m041", "channel_id": "C005", "user_name": "carol", "text": "Performance reviews are due May 31st. Please complete your self-assessments in Lattice before then. Reach out to HR with any questions.", "timestamp": "2026-04-30T10:00:00Z", "is_pinned": False, "reply_count": 0},
    {"id": "m042", "channel_id": "C005", "user_name": "alice", "text": "We're moving to a 4-day work week pilot starting June 1st! More details in the all-hands deck.", "timestamp": "2026-05-01T11:00:00Z", "is_pinned": True, "reply_count": 0},
]

_SEED_REACTIONS = [
    {"id": "rx001", "message_id": "m001", "emoji": "👋", "user_name": "bob"},
    {"id": "rx002", "message_id": "m001", "emoji": "👋", "user_name": "carol"},
    {"id": "rx003", "message_id": "m001", "emoji": "👋", "user_name": "me"},
    {"id": "rx004", "message_id": "m003", "emoji": "🎉", "user_name": "alice"},
    {"id": "rx005", "message_id": "m003", "emoji": "🎉", "user_name": "bob"},
    {"id": "rx006", "message_id": "m003", "emoji": "🎉", "user_name": "dave"},
    {"id": "rx007", "message_id": "m003", "emoji": "👏", "user_name": "me"},
    {"id": "rx008", "message_id": "m004", "emoji": "👍", "user_name": "alice"},
    {"id": "rx009", "message_id": "m004", "emoji": "👍", "user_name": "eve"},
    {"id": "rx010", "message_id": "m007", "emoji": "🚀", "user_name": "bob"},
    {"id": "rx011", "message_id": "m007", "emoji": "🚀", "user_name": "carol"},
    {"id": "rx012", "message_id": "m007", "emoji": "🚀", "user_name": "me"},
    {"id": "rx013", "message_id": "m010", "emoji": "👀", "user_name": "alice"},
    {"id": "rx014", "message_id": "m010", "emoji": "👀", "user_name": "bob"},
    {"id": "rx015", "message_id": "m012", "emoji": "🙌", "user_name": "carol"},
    {"id": "rx016", "message_id": "m012", "emoji": "🙌", "user_name": "dave"},
    {"id": "rx017", "message_id": "m014", "emoji": "⚡", "user_name": "alice"},
    {"id": "rx018", "message_id": "m014", "emoji": "⚡", "user_name": "eve"},
    {"id": "rx019", "message_id": "m030", "emoji": "😂", "user_name": "alice"},
    {"id": "rx020", "message_id": "m030", "emoji": "😂", "user_name": "carol"},
    {"id": "rx021", "message_id": "m030", "emoji": "🔥", "user_name": "dave"},
    {"id": "rx022", "message_id": "m040", "emoji": "🎉", "user_name": "alice"},
    {"id": "rx023", "message_id": "m040", "emoji": "🎉", "user_name": "bob"},
    {"id": "rx024", "message_id": "m040", "emoji": "🎉", "user_name": "carol"},
    {"id": "rx025", "message_id": "m040", "emoji": "🎉", "user_name": "dave"},
    {"id": "rx026", "message_id": "m040", "emoji": "🎉", "user_name": "eve"},
    {"id": "rx027", "message_id": "m042", "emoji": "🙌", "user_name": "bob"},
    {"id": "rx028", "message_id": "m042", "emoji": "🙌", "user_name": "carol"},
    {"id": "rx029", "message_id": "m042", "emoji": "🙌", "user_name": "me"},
]

_SEED_DMS = [
    {"id": "dm001", "from_user": "alice", "to_user": "me", "text": "Hey, do you have a minute to chat about the Q3 roadmap presentation?", "timestamp": "2026-05-04T11:30:00Z", "is_read": False},
    {"id": "dm002", "from_user": "alice", "to_user": "me", "text": "I want to make sure we're aligned before the all-hands.", "timestamp": "2026-05-04T11:31:00Z", "is_read": False},
    {"id": "dm003", "from_user": "me", "to_user": "alice", "text": "Sure! Give me 10 mins, wrapping up a PR review.", "timestamp": "2026-05-04T11:35:00Z", "is_read": True},
    {"id": "dm004", "from_user": "bob", "to_user": "me", "text": "Thanks for the quick review on PR #312! Your feedback was really helpful.", "timestamp": "2026-05-03T16:05:00Z", "is_read": True},
    {"id": "dm005", "from_user": "me", "to_user": "bob", "text": "Of course! The session management fix was the right call. Looks solid now.", "timestamp": "2026-05-03T16:10:00Z", "is_read": True},
    {"id": "dm006", "from_user": "bob", "to_user": "me", "text": "Want to grab lunch tomorrow and sync on the sprint?", "timestamp": "2026-05-03T16:15:00Z", "is_read": True},
    {"id": "dm007", "from_user": "carol", "to_user": "me", "text": "Quick question — can you fill out the skills matrix for the team skills audit? Link: https://forms.acme.com/skills", "timestamp": "2026-05-02T14:00:00Z", "is_read": False},
    {"id": "dm008", "from_user": "carol", "to_user": "me", "text": "Deadline is this Friday. Should only take 10 minutes!", "timestamp": "2026-05-02T14:01:00Z", "is_read": False},
]

_SEED_READ_STATES = [
    {"channel_id": "C001", "unread_count": 2},
    {"channel_id": "C002", "unread_count": 1},
    {"channel_id": "C003", "unread_count": 3},
    {"channel_id": "C004", "unread_count": 4},
    {"channel_id": "C005", "unread_count": 0},
]


def _seed_if_empty() -> None:
    with SessionLocal() as db:
        if db.query(Channel).count() > 0:
            return
        for c in _SEED_CHANNELS:
            db.add(Channel(
                id=c["id"], name=c["name"], purpose=c["purpose"],
                is_private=c["is_private"], is_archived=c["is_archived"],
                created_at="2026-01-01T00:00:00Z",
            ))
        for m in _SEED_MESSAGES:
            db.add(Message(
                id=m["id"], channel_id=m["channel_id"], user_name=m["user_name"],
                text=m["text"], timestamp=m["timestamp"], is_pinned=m["is_pinned"],
                thread_parent_id=None, reply_count=m["reply_count"],
            ))
        for r in _SEED_REACTIONS:
            db.add(Reaction(id=r["id"], message_id=r["message_id"], emoji=r["emoji"], user_name=r["user_name"]))
        for dm in _SEED_DMS:
            db.add(DirectMessage(
                id=dm["id"], from_user=dm["from_user"], to_user=dm["to_user"],
                text=dm["text"], timestamp=dm["timestamp"], is_read=dm["is_read"],
            ))
        db.add(UserStatus(user_name="me", status_text="", status_emoji="", presence="online"))
        for rs in _SEED_READ_STATES:
            db.add(ChannelReadState(channel_id=rs["channel_id"], unread_count=rs["unread_count"]))
        db.commit()


_seed_if_empty()


# ---------------------------------------------------------------------------
# Forge protocol
# ---------------------------------------------------------------------------

@app.get("/forge/health")
def health():
    return {"status": "ok"}


@app.get("/forge/state")
def forge_state():
    with SessionLocal() as db:
        return _get_state_dict(db)


@app.post("/forge/reset")
def forge_reset():
    with SessionLocal() as db:
        db.query(Reaction).delete()
        db.query(Message).delete()
        db.query(Channel).delete()
        db.query(DirectMessage).delete()
        db.query(UserStatus).delete()
        db.query(ChannelReadState).delete()
        db.query(SavedState).delete()
        db.commit()
    _seed_if_empty()
    with SessionLocal() as db:
        return {"status": "reset", "state": _get_state_dict(db)}


class SnapshotRequest(BaseModel):
    slot: str


@app.post("/forge/snapshot")
def forge_snapshot(req: SnapshotRequest):
    with SessionLocal() as db:
        data = _dump_full_db(db)
        saved = db.query(SavedState).filter(SavedState.slot == req.slot).first()
        if saved:
            saved.data = json.dumps(data)
        else:
            db.add(SavedState(slot=req.slot, data=json.dumps(data)))
        db.commit()
    return {"status": "snapshot_saved", "slot": req.slot}


@app.post("/forge/restore/{slot}")
def forge_restore(slot: str):
    with SessionLocal() as db:
        saved = db.query(SavedState).filter(SavedState.slot == slot).first()
        if not saved:
            raise HTTPException(status_code=404, detail=f"Slot '{slot}' not found")
        data = json.loads(saved.data)
        _restore_from_dict(db, data)
        state = _get_state_dict(db)
    return {"status": "restored", "slot": slot, "state": state}


@app.post("/forge/restore-state")
def forge_restore_state(data: dict):
    with SessionLocal() as db:
        _restore_from_dict(db, data)
        state = _get_state_dict(db)
    return {"status": "restored", "state": state}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/ui")
def ui():
    return FileResponse("ui.html")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SendMessageRequest(BaseModel):
    channel: str
    text: str


class ReplyThreadRequest(BaseModel):
    channel: str
    message_id: str
    text: str


class ReactionRequest(BaseModel):
    channel: str
    message_id: str
    emoji: str


class PinRequest(BaseModel):
    channel: str
    message_id: str


class DeleteMessageRequest(BaseModel):
    channel: str
    message_id: str


class CreateChannelRequest(BaseModel):
    name: str
    purpose: Optional[str] = ""


class ArchiveChannelRequest(BaseModel):
    channel: str


class SetStatusRequest(BaseModel):
    status: str
    emoji: Optional[str] = ""


class SendDmRequest(BaseModel):
    to: str
    text: str


class MarkChannelReadRequest(BaseModel):
    channel: str


class SearchMessagesRequest(BaseModel):
    query: str
    channel: Optional[str] = None


class GetChannelMessagesRequest(BaseModel):
    channel: str
    limit: Optional[int] = 50


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _get_channel(db: Session, channel_ref: str) -> Channel | None:
    return db.query(Channel).filter(
        (Channel.id == channel_ref) | (Channel.name == channel_ref)
    ).first()


@app.post("/send_message")
def send_message(req: SendMessageRequest):
    with SessionLocal() as db:
        channel = _get_channel(db, req.channel)
        if not channel:
            return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _get_state_dict(db)}
        if channel.is_archived:
            return {"status": "error", "message": f"Channel '{req.channel}' is archived", "state": _get_state_dict(db)}
        msg_id = f"m{uuid.uuid4().hex[:8]}"
        db.add(Message(
            id=msg_id, channel_id=channel.id, user_name="me",
            text=req.text, timestamp=_now(), is_pinned=False,
            thread_parent_id=None, reply_count=0,
        ))
        db.commit()
        state = _get_state_dict(db)
    return {"status": "sent", "message_id": msg_id, "channel": req.channel, "state": state}


@app.post("/reply_thread")
def reply_thread(req: ReplyThreadRequest):
    with SessionLocal() as db:
        channel = _get_channel(db, req.channel)
        if not channel:
            return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _get_state_dict(db)}
        parent = db.query(Message).filter(Message.id == req.message_id, Message.channel_id == channel.id).first()
        if not parent:
            return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _get_state_dict(db)}
        reply_id = f"t{uuid.uuid4().hex[:8]}"
        db.add(Message(
            id=reply_id, channel_id=channel.id, user_name="me",
            text=req.text, timestamp=_now(), is_pinned=False,
            thread_parent_id=req.message_id, reply_count=0,
        ))
        parent.reply_count += 1
        db.commit()
        state = _get_state_dict(db)
    return {"status": "replied", "reply_id": reply_id, "thread_of": req.message_id, "state": state}


@app.post("/add_reaction")
def add_reaction(req: ReactionRequest):
    with SessionLocal() as db:
        channel = _get_channel(db, req.channel)
        if not channel:
            return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _get_state_dict(db)}
        msg = db.query(Message).filter(Message.id == req.message_id, Message.channel_id == channel.id).first()
        if not msg:
            return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _get_state_dict(db)}
        existing = db.query(Reaction).filter(
            Reaction.message_id == req.message_id,
            Reaction.emoji == req.emoji,
            Reaction.user_name == "me",
        ).first()
        if not existing:
            db.add(Reaction(
                id=f"rx{uuid.uuid4().hex[:8]}",
                message_id=req.message_id, emoji=req.emoji, user_name="me",
            ))
            db.commit()
        state = _get_state_dict(db)
    return {"status": "reacted", "emoji": req.emoji, "message_id": req.message_id, "state": state}


@app.post("/remove_reaction")
def remove_reaction(req: ReactionRequest):
    with SessionLocal() as db:
        existing = db.query(Reaction).filter(
            Reaction.message_id == req.message_id,
            Reaction.emoji == req.emoji,
            Reaction.user_name == "me",
        ).first()
        if existing:
            db.delete(existing)
            db.commit()
        state = _get_state_dict(db)
    return {"status": "removed", "emoji": req.emoji, "message_id": req.message_id, "state": state}


@app.post("/pin_message")
def pin_message(req: PinRequest):
    with SessionLocal() as db:
        channel = _get_channel(db, req.channel)
        if not channel:
            return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _get_state_dict(db)}
        msg = db.query(Message).filter(Message.id == req.message_id, Message.channel_id == channel.id).first()
        if not msg:
            return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _get_state_dict(db)}
        msg.is_pinned = True
        db.commit()
        state = _get_state_dict(db)
    return {"status": "pinned", "message_id": req.message_id, "state": state}


@app.post("/unpin_message")
def unpin_message(req: PinRequest):
    with SessionLocal() as db:
        channel = _get_channel(db, req.channel)
        if not channel:
            return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _get_state_dict(db)}
        msg = db.query(Message).filter(Message.id == req.message_id, Message.channel_id == channel.id).first()
        if not msg:
            return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _get_state_dict(db)}
        msg.is_pinned = False
        db.commit()
        state = _get_state_dict(db)
    return {"status": "unpinned", "message_id": req.message_id, "state": state}


@app.post("/delete_message")
def delete_message(req: DeleteMessageRequest):
    with SessionLocal() as db:
        channel = _get_channel(db, req.channel)
        if not channel:
            return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _get_state_dict(db)}
        msg = db.query(Message).filter(Message.id == req.message_id, Message.channel_id == channel.id).first()
        if not msg:
            return {"status": "error", "message": f"Message '{req.message_id}' not found", "state": _get_state_dict(db)}
        db.query(Reaction).filter(Reaction.message_id == req.message_id).delete()
        db.delete(msg)
        db.commit()
        state = _get_state_dict(db)
    return {"status": "deleted", "message_id": req.message_id, "state": state}


@app.post("/create_channel")
def create_channel(req: CreateChannelRequest):
    with SessionLocal() as db:
        existing = _get_channel(db, req.name)
        if existing:
            return {"status": "error", "message": f"Channel '{req.name}' already exists", "state": _get_state_dict(db)}
        channel_id = f"C{uuid.uuid4().hex[:6].upper()}"
        db.add(Channel(
            id=channel_id,
            name=req.name.lower().replace(" ", "-"),
            purpose=req.purpose or "",
            is_private=False,
            is_archived=False,
            created_at=_now(),
        ))
        db.add(ChannelReadState(channel_id=channel_id, unread_count=0))
        db.commit()
        state = _get_state_dict(db)
    return {"status": "created", "channel_id": channel_id, "name": req.name, "state": state}


@app.post("/archive_channel")
def archive_channel(req: ArchiveChannelRequest):
    with SessionLocal() as db:
        channel = _get_channel(db, req.channel)
        if not channel:
            return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _get_state_dict(db)}
        channel.is_archived = True
        db.commit()
        state = _get_state_dict(db)
    return {"status": "archived", "channel": req.channel, "state": state}


@app.post("/set_status")
def set_status(req: SetStatusRequest):
    with SessionLocal() as db:
        status = db.query(UserStatus).filter(UserStatus.user_name == "me").first()
        if status:
            status.status_text = req.status
            status.status_emoji = req.emoji or ""
        else:
            db.add(UserStatus(user_name="me", status_text=req.status, status_emoji=req.emoji or "", presence="online"))
        db.commit()
        state = _get_state_dict(db)
    return {"status": "updated", "new_status": req.status, "emoji": req.emoji, "state": state}


@app.post("/send_dm")
def send_dm(req: SendDmRequest):
    with SessionLocal() as db:
        dm_id = f"dm{uuid.uuid4().hex[:8]}"
        db.add(DirectMessage(
            id=dm_id, from_user="me", to_user=req.to,
            text=req.text, timestamp=_now(), is_read=True,
        ))
        db.commit()
        state = _get_state_dict(db)
    return {"status": "sent", "dm_id": dm_id, "to": req.to, "state": state}


@app.post("/mark_channel_read")
def mark_channel_read(req: MarkChannelReadRequest):
    with SessionLocal() as db:
        channel = _get_channel(db, req.channel)
        if not channel:
            return {"status": "error", "message": f"Channel '{req.channel}' not found", "state": _get_state_dict(db)}
        rs = db.query(ChannelReadState).filter(ChannelReadState.channel_id == channel.id).first()
        if rs:
            rs.unread_count = 0
        db.commit()
        state = _get_state_dict(db)
    return {"status": "marked_read", "channel": req.channel, "state": state}


@app.post("/search_messages")
def search_messages(req: SearchMessagesRequest):
    q = req.query.lower()
    with SessionLocal() as db:
        query = db.query(Message)
        if req.channel:
            channel = _get_channel(db, req.channel)
            if channel:
                query = query.filter(Message.channel_id == channel.id)
        messages = query.all()
        results = [_message_to_dict(m, db) for m in messages if q in m.text.lower() or q in m.user_name.lower()]
        state = _get_state_dict(db)
    return {"results": results, "count": len(results), "state": state}


@app.post("/get_channel_messages")
@app.get("/get_channel_messages")
def get_channel_messages(req: GetChannelMessagesRequest = None, channel: str = None, limit: int = 50):
    # Support both POST (with body) and GET (with query params)
    chan_ref = req.channel if req else channel
    lim = req.limit if req and req.limit else limit
    if not chan_ref:
        return {"status": "error", "message": "channel parameter required"}
    with SessionLocal() as db:
        chan = _get_channel(db, chan_ref)
        if not chan:
            return {"status": "error", "message": f"Channel '{chan_ref}' not found"}
        messages = (
            db.query(Message)
            .filter(Message.channel_id == chan.id, Message.thread_parent_id == None)  # noqa: E711
            .order_by(Message.timestamp)
            .limit(lim)
            .all()
        )
        result = [_message_to_dict(m, db) for m in messages]
    return {"channel": chan_ref, "messages": result, "count": len(result)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
