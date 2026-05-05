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


class ActionLog(Base):
    __tablename__ = "action_log"
    id = Column(String, primary_key=True, default=lambda: f"a{uuid.uuid4().hex[:8]}")
    action_type = Column(String, nullable=False)
    target_id = Column(String, nullable=True)
    payload = Column(Text, default="{}")
    timestamp = Column(String, nullable=False)


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


def _log_action(db: Session, action_type: str, target_id: str = None, payload: dict = None) -> None:
    db.add(ActionLog(
        id=f"a{uuid.uuid4().hex[:8]}",
        action_type=action_type,
        target_id=target_id,
        payload=json.dumps(payload or {}),
        timestamp=_now(),
    ))


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
        "actions_taken": db.query(ActionLog).count(),
        "recent_actions": [
            {"id": a.id, "action_type": a.action_type, "target_id": a.target_id, "timestamp": a.timestamp}
            for a in db.query(ActionLog).order_by(ActionLog.timestamp.desc()).limit(10).all()
        ],
        "pinned_messages_deleted": db.query(ActionLog).filter(
            ActionLog.action_type == "delete_message",
            ActionLog.payload.contains('"was_pinned": true')
        ).count(),
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
    action_log = [
        {"id": a.id, "action_type": a.action_type, "target_id": a.target_id,
         "payload": a.payload, "timestamp": a.timestamp}
        for a in db.query(ActionLog).order_by(ActionLog.timestamp).all()
    ]
    return {
        "channels": channels, "messages": messages, "reactions": reactions,
        "direct_messages": dms, "user_statuses": statuses, "read_states": read_states,
        "action_log": action_log,
    }


def _restore_from_dict(db: Session, data: dict) -> None:
    db.query(Reaction).delete()
    db.query(Message).delete()
    db.query(Channel).delete()
    db.query(DirectMessage).delete()
    db.query(UserStatus).delete()
    db.query(ChannelReadState).delete()
    db.query(ActionLog).delete()
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
    for a in data.get("action_log", []):
        db.add(ActionLog(id=a["id"], action_type=a["action_type"], target_id=a.get("target_id"),
                         payload=a.get("payload", "{}"), timestamp=a["timestamp"]))
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
    {"id": "C006", "name": "design", "purpose": "Design system, Figma files, and UI/UX discussions", "is_private": False, "is_archived": False},
    {"id": "C007", "name": "ops-infra", "purpose": "Infrastructure, AWS costs, Kubernetes, and monitoring", "is_private": False, "is_archived": False},
]

_SEED_MESSAGES = [
    # ---- #general ----
    {"id": "m001", "channel_id": "C001", "user_name": "alice", "text": "Good morning team! Reminder: company all-hands is today at 3pm on Zoom. Check your calendar for the link. We'll cover Q2 results, the Northstar deal, and preview Q3 priorities.", "timestamp": "2026-05-04T08:00:00Z", "is_pinned": True, "reply_count": 2},
    {"id": "m002", "channel_id": "C001", "user_name": "carol", "text": "Q2 results are in and they're incredible — we hit 118% of our revenue target! The entire company crushed it. Special shoutout to sales and engineering 🎉", "timestamp": "2026-05-04T09:30:00Z", "is_pinned": True, "reply_count": 4},
    {"id": "m003", "channel_id": "C001", "user_name": "eve", "text": "The new onboarding docs are live at docs.acme.com/onboarding. Spent the whole week writing and polishing them — covers environment setup, the agent protocol, and common troubleshooting. Feedback welcome!", "timestamp": "2026-05-03T16:45:00Z", "is_pinned": False, "reply_count": 2},
    {"id": "m004", "channel_id": "C001", "user_name": "dave", "text": "Heads-up: deploying the new metrics dashboard at 5pm EST. Expect ~5 min downtime on the reporting page while the migration runs. Will post here when it's back up.", "timestamp": "2026-05-04T14:00:00Z", "is_pinned": False, "reply_count": 3},
    {"id": "m005", "channel_id": "C001", "user_name": "me", "text": "Quick heads-up: I'll be OOO this Friday. Moved my only Friday meeting and asked @alice to cover anything urgent. Will be back and responsive Monday morning.", "timestamp": "2026-05-04T15:30:00Z", "is_pinned": False, "reply_count": 0},
    {"id": "m006", "channel_id": "C001", "user_name": "alice", "text": "Huge news 🎉 We just signed our biggest enterprise deal ever — Northstar Capital is joining as a customer! This has been months in the making. More details in today's all-hands.", "timestamp": "2026-05-02T11:00:00Z", "is_pinned": True, "reply_count": 5},
    {"id": "m007", "channel_id": "C001", "user_name": "bob", "text": "My IDE just autocompleted a function I was about to write, named it exactly what I was going to name it, and got the logic right. I'm either psychic or we trained a model on my commit history.", "timestamp": "2026-05-01T14:00:00Z", "is_pinned": False, "reply_count": 0},

    # ---- #engineering ----
    {"id": "m010", "channel_id": "C002", "user_name": "eve", "text": "PR #312 is ready for review: https://github.com/acme/forge/pull/312 — adds the SQLAlchemy persistence layer. ~400 LOC, all 47 unit tests passing. Main changes: session management, connection pooling, migration script. Would love eyes on the rollback path.", "timestamp": "2026-05-04T10:00:00Z", "is_pinned": False, "reply_count": 4},
    {"id": "m011", "channel_id": "C002", "user_name": "alice", "text": "Heads up: upgrading to Python 3.12 after the v2.4 release. Please audit your dependencies for compatibility issues, especially anything with C extensions. I'll send a migration guide follow-up.", "timestamp": "2026-05-04T11:00:00Z", "is_pinned": True, "reply_count": 2},
    {"id": "m012", "channel_id": "C002", "user_name": "bob", "text": "The flaky CI test has been slain! 🎯 Root cause: race condition in auth token refresh — two workers refreshing simultaneously, second one got a stale token. Fixed with a per-session mutex. 3 hours well spent.", "timestamp": "2026-05-04T16:00:00Z", "is_pinned": False, "reply_count": 3},
    {"id": "m013", "channel_id": "C002", "user_name": "carol", "text": "Post-mortem for last week's DB spike is live: https://docs.acme.com/postmortem-042 — please add comments and action items by EOD tomorrow. Want to close this before the sprint ends.", "timestamp": "2026-05-03T15:00:00Z", "is_pinned": False, "reply_count": 2},
    {"id": "m014", "channel_id": "C002", "user_name": "me", "text": "Just merged the async worker refactor. Early metrics: background job latency down 60% in staging. Rolling out to prod now and monitoring in Grafana. Ping me if you see anything weird.", "timestamp": "2026-05-03T17:30:00Z", "is_pinned": False, "reply_count": 2},
    {"id": "m015", "channel_id": "C002", "user_name": "dave", "text": "Code freeze reminder: v2.4.0 cuts Thursday at noon. After that it's hotfixes only until the release ships Friday. Please get in-flight PRs merged or move them to the v2.5 milestone.", "timestamp": "2026-05-02T09:00:00Z", "is_pinned": True, "reply_count": 0},
    {"id": "m016", "channel_id": "C002", "user_name": "bob", "text": "Evaluating ClickHouse to replace our analytics Postgres. Main use case: time-series aggregations on agent sessions — billions of rows, heavy GROUP BY and window functions. Anyone used it at scale? What should I watch out for?", "timestamp": "2026-05-01T14:00:00Z", "is_pinned": False, "reply_count": 5},
    {"id": "m017", "channel_id": "C002", "user_name": "alice", "text": "New hire Sarah Chen starts May 12th as VP Engineering — joining from Google infrastructure with deep distributed systems experience. She'll be in engineering standup from day 1. Please make her feel welcome!", "timestamp": "2026-05-04T11:00:00Z", "is_pinned": False, "reply_count": 2},

    # ---- #product ----
    {"id": "m020", "channel_id": "C003", "user_name": "carol", "text": "Updated the Q3 roadmap in Notion — big additions: AI-powered semantic search, bulk actions for agent management, redesigned onboarding flow, and SSO for enterprise. Link: https://notion.so/acme/q3-roadmap. Please review and comment!", "timestamp": "2026-05-04T10:30:00Z", "is_pinned": False, "reply_count": 6},
    {"id": "m021", "channel_id": "C003", "user_name": "dave", "text": "Mocked up 3 dashboard options: Option A is a bold side-panel layout, Option B is familiar top-nav with improved cards, Option C is minimal command-palette-first. Figma: https://figma.com/file/acme/dashboard-v2. Which do you prefer?", "timestamp": "2026-05-04T13:00:00Z", "is_pinned": False, "reply_count": 4},
    {"id": "m022", "channel_id": "C003", "user_name": "me", "text": "Wrapped user research calls with 5 enterprise customers. Main themes: (1) episode resets too slow — avg 40s, want sub-10s, (2) no bulk actions makes managing 20+ agents painful, (3) observability tab is the most-loved feature. Summarizing in a doc.", "timestamp": "2026-05-03T11:00:00Z", "is_pinned": False, "reply_count": 2},
    {"id": "m023", "channel_id": "C003", "user_name": "alice", "text": "NPS score this month: 72 🎉 Up from 61 last month! The single biggest driver in qualitative responses was snapshot/restore. Great validation of the work the team has been doing.", "timestamp": "2026-05-02T16:00:00Z", "is_pinned": True, "reply_count": 3},
    {"id": "m024", "channel_id": "C003", "user_name": "carol", "text": "Quick question: where's the canonical documentation for the agent observation format? A customer asked me today and I couldn't find it in the public docs. Trying to write the API reference section.", "timestamp": "2026-05-01T13:45:00Z", "is_pinned": False, "reply_count": 1},

    # ---- #random ----
    {"id": "m030", "channel_id": "C004", "user_name": "bob", "text": "Hot take: tabs are better than spaces and I will not be taking questions at this time", "timestamp": "2026-05-04T12:00:00Z", "is_pinned": False, "reply_count": 7},
    {"id": "m031", "channel_id": "C004", "user_name": "carol", "text": "Anyone watching the playoff game tonight? 🏀 Thinking we could do an impromptu watch party in the office — big TV in the breakroom. Who's in?", "timestamp": "2026-05-04T13:00:00Z", "is_pinned": False, "reply_count": 4},
    {"id": "m032", "channel_id": "C004", "user_name": "dave", "text": "Just discovered you can ask Claude to play chess via the terminal and it knows real openings, sacrifices pieces for positional pressure, and talks trash mid-game. What is this timeline we're living in.", "timestamp": "2026-05-04T14:30:00Z", "is_pinned": False, "reply_count": 3},
    {"id": "m033", "channel_id": "C004", "user_name": "eve", "text": "Friendly reminder: I said I'd buy tacos for everyone if we hit the sprint goal. We still have 3 story points left. Friday is coming. No pressure 👀🌮", "timestamp": "2026-05-03T17:00:00Z", "is_pinned": False, "reply_count": 5},
    {"id": "m034", "channel_id": "C004", "user_name": "me", "text": "We hit the sprint goal! All 3 remaining points merged this morning. Tacos on Eve 🌮🌮🌮 See everyone at 12:30pm Friday.", "timestamp": "2026-05-03T18:00:00Z", "is_pinned": False, "reply_count": 0},
    {"id": "m035", "channel_id": "C004", "user_name": "alice", "text": "Remote work tip: I keep a sticky note on my monitor that says 'You are not your Slack response time.' Genuinely changed my anxiety levels.", "timestamp": "2026-05-02T10:00:00Z", "is_pinned": False, "reply_count": 0},

    # ---- #announcements ----
    {"id": "m040", "channel_id": "C005", "user_name": "alice", "text": "Welcome to our new VP of Engineering, Sarah Chen! 🎉 Sarah joins from Google where she led a 60-person infrastructure team. Deep experience in distributed systems, SRE, and engineering culture. She starts May 12th — please give her a warm welcome!", "timestamp": "2026-04-28T09:00:00Z", "is_pinned": True, "reply_count": 0},
    {"id": "m041", "channel_id": "C005", "user_name": "carol", "text": "Reminder: performance reviews are due May 31st. Please complete your self-assessment in Lattice before then. Manager reviews open June 1st. Reach out to HR with any questions.", "timestamp": "2026-04-30T10:00:00Z", "is_pinned": False, "reply_count": 0},
    {"id": "m042", "channel_id": "C005", "user_name": "alice", "text": "Exciting: starting June 1st we're piloting a 4-day workweek for the summer. No change to pay. Team voted for Fridays off. Full details and FAQ in the all-hands deck at docs.acme.com/all-hands-may.", "timestamp": "2026-05-01T11:00:00Z", "is_pinned": True, "reply_count": 0},

    # ---- #design ----
    {"id": "m050", "channel_id": "C006", "user_name": "dave", "text": "Design system kickoff! 🎨 Building a shared component library in Storybook. Goal: every UI component in one place with documented variants, states, and accessibility notes. Starting with Button, Input, Modal, and Card this sprint. Figma master file is pinned in the channel.", "timestamp": "2026-05-04T10:30:00Z", "is_pinned": False, "reply_count": 3},
    {"id": "m051", "channel_id": "C006", "user_name": "carol", "text": "Component library feedback from my review: hover states on Button are inconsistent — primary uses 5% lighter background, secondary uses 10% darker border. Let's standardize. Also the Modal close button is missing a focus ring. Annotated screenshots in thread.", "timestamp": "2026-05-04T11:30:00Z", "is_pinned": False, "reply_count": 2},
    {"id": "m052", "channel_id": "C006", "user_name": "eve", "text": "Found a bunch of layout issues at <768px: sidebar nav overlaps content area, data table doesn't scroll horizontally, and the action bar buttons wrap to 3 lines. Creating Jira tickets for each but flagging here for visibility.", "timestamp": "2026-05-04T13:15:00Z", "is_pinned": False, "reply_count": 2},
    {"id": "m053", "channel_id": "C006", "user_name": "me", "text": "Proposing a color palette update: swapping our blue (#1a6ef5) for a warmer variant (#2563eb) to better match brand guidelines. Also adding success green (#16a34a) and warning amber (#d97706) — both pass WCAG AA. Figma prototype attached.", "timestamp": "2026-05-04T14:30:00Z", "is_pinned": False, "reply_count": 1},
    {"id": "m054", "channel_id": "C006", "user_name": "dave", "text": "Running the WCAG accessibility audit this week on core pages. Will share findings before sprint end. If you've touched any forms or interactive elements recently, please do a quick tab-navigation test.", "timestamp": "2026-05-03T09:00:00Z", "is_pinned": False, "reply_count": 0},

    # ---- #ops-infra ----
    {"id": "m060", "channel_id": "C007", "user_name": "alice", "text": "Alert: AWS costs spiked 34% vs last week. We're at 78% of the monthly budget with 12 days left. Need eyes on this ASAP. Tagging @bob and @me — can someone dig into CloudWatch and find the culprit?", "timestamp": "2026-05-04T10:00:00Z", "is_pinned": True, "reply_count": 4},
    {"id": "m061", "channel_id": "C007", "user_name": "bob", "text": "Planning the Kubernetes upgrade from 1.27 to 1.28 next maintenance window (Sunday 2am). Key changes: better pod scheduling, improved HPA behavior, removal of some deprecated beta APIs. Sending a runbook Friday. Anyone want to pair on the upgrade?", "timestamp": "2026-05-04T11:00:00Z", "is_pinned": False, "reply_count": 3},
    {"id": "m062", "channel_id": "C007", "user_name": "me", "text": "New monitoring dashboard is live in Grafana: tracks action latency p50/p99, active agent sessions, queue depth, and error rates by endpoint. PagerDuty alerts set for p99 > 2s sustained 5 min. Link in channel topic.", "timestamp": "2026-05-04T12:00:00Z", "is_pinned": False, "reply_count": 2},
    {"id": "m063", "channel_id": "C007", "user_name": "carol", "text": "Monthly backup verification complete — all 7 critical DBs restored successfully in staging. RTO for the largest DB (agents) is ~8 minutes. Documenting in the runbooks.", "timestamp": "2026-05-03T16:00:00Z", "is_pinned": False, "reply_count": 0},
]

_SEED_REACTIONS = [
    {"id": "rx001", "message_id": "m001", "emoji": "👋", "user_name": "bob"},
    {"id": "rx002", "message_id": "m001", "emoji": "👋", "user_name": "carol"},
    {"id": "rx003", "message_id": "m001", "emoji": "👋", "user_name": "me"},
    {"id": "rx004", "message_id": "m002", "emoji": "🎉", "user_name": "alice"},
    {"id": "rx005", "message_id": "m002", "emoji": "🎉", "user_name": "bob"},
    {"id": "rx006", "message_id": "m002", "emoji": "🎉", "user_name": "dave"},
    {"id": "rx007", "message_id": "m002", "emoji": "👏", "user_name": "me"},
    {"id": "rx008", "message_id": "m004", "emoji": "👍", "user_name": "alice"},
    {"id": "rx009", "message_id": "m004", "emoji": "👍", "user_name": "eve"},
    {"id": "rx010", "message_id": "m006", "emoji": "🚀", "user_name": "bob"},
    {"id": "rx011", "message_id": "m006", "emoji": "🚀", "user_name": "carol"},
    {"id": "rx012", "message_id": "m006", "emoji": "🚀", "user_name": "me"},
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
    {"id": "rx030", "message_id": "m020", "emoji": "👀", "user_name": "alice"},
    {"id": "rx031", "message_id": "m020", "emoji": "👀", "user_name": "bob"},
    {"id": "rx032", "message_id": "m023", "emoji": "🎉", "user_name": "bob"},
    {"id": "rx033", "message_id": "m023", "emoji": "🎉", "user_name": "dave"},
    {"id": "rx034", "message_id": "m060", "emoji": "😬", "user_name": "bob"},
    {"id": "rx035", "message_id": "m060", "emoji": "😬", "user_name": "carol"},
    {"id": "rx036", "message_id": "m062", "emoji": "🙌", "user_name": "alice"},
    {"id": "rx037", "message_id": "m062", "emoji": "🙌", "user_name": "dave"},
    {"id": "rx038", "message_id": "m050", "emoji": "🎨", "user_name": "carol"},
    {"id": "rx039", "message_id": "m050", "emoji": "🎨", "user_name": "eve"},
    {"id": "rx040", "message_id": "m034", "emoji": "🌮", "user_name": "alice"},
    {"id": "rx041", "message_id": "m034", "emoji": "🌮", "user_name": "bob"},
    {"id": "rx042", "message_id": "m034", "emoji": "🌮", "user_name": "carol"},
    {"id": "rx043", "message_id": "m034", "emoji": "🌮", "user_name": "dave"},
]

_SEED_DMS = [
    {"id": "dm001", "from_user": "alice", "to_user": "me", "text": "Hey, do you have a minute to chat about the Q3 roadmap before the all-hands? Want to make sure we're aligned on the key priorities.", "timestamp": "2026-05-04T11:30:00Z", "is_read": False},
    {"id": "dm002", "from_user": "alice", "to_user": "me", "text": "Specifically the AI search timeline — engineering asked me about it and I want to give them the right answer.", "timestamp": "2026-05-04T11:31:00Z", "is_read": False},
    {"id": "dm003", "from_user": "me", "to_user": "alice", "text": "Sure! Give me 10 mins, wrapping up a PR review. Can hop on a quick call after.", "timestamp": "2026-05-04T11:35:00Z", "is_read": True},
    {"id": "dm004", "from_user": "bob", "to_user": "me", "text": "Thanks for the thorough review on PR #312! The feedback on exception handling was exactly right — I had been too permissive with bare except clauses.", "timestamp": "2026-05-03T16:05:00Z", "is_read": True},
    {"id": "dm005", "from_user": "me", "to_user": "bob", "text": "Of course! The session management approach is solid. The rollback test you added should catch any future regressions there.", "timestamp": "2026-05-03T16:10:00Z", "is_read": True},
    {"id": "dm006", "from_user": "bob", "to_user": "me", "text": "Want to grab lunch tomorrow and sync on the ClickHouse evaluation? I'm leaning toward DuckDB after reading the thread but want to talk it through.", "timestamp": "2026-05-03T16:15:00Z", "is_read": True},
    {"id": "dm007", "from_user": "carol", "to_user": "me", "text": "Quick ask — can you fill out the skills matrix for the team skills audit? Link: https://forms.acme.com/skills. It feeds into the Q3 hiring plan.", "timestamp": "2026-05-02T14:00:00Z", "is_read": False},
    {"id": "dm008", "from_user": "carol", "to_user": "me", "text": "Deadline is this Friday. Should take about 10 minutes. Really appreciate it!", "timestamp": "2026-05-02T14:01:00Z", "is_read": False},
    {"id": "dm009", "from_user": "dave", "to_user": "me", "text": "Deploy went smoothly! Dashboard is live. Metrics look clean — no errors in the first 30 minutes. Thanks for keeping an eye on Grafana during the window.", "timestamp": "2026-05-04T17:30:00Z", "is_read": True},
    {"id": "dm010", "from_user": "me", "to_user": "dave", "text": "Great news! The latency improvement from the async refactor is holding up in prod too. Really good day for the infra.", "timestamp": "2026-05-04T17:35:00Z", "is_read": True},
    {"id": "dm011", "from_user": "eve", "to_user": "me", "text": "Hey! Quick request — can you sanity-check my design system PR? It's just the Button component. Should be a 5-minute review, I just want another set of eyes before I merge.", "timestamp": "2026-05-04T16:15:00Z", "is_read": False},
    {"id": "dm012", "from_user": "eve", "to_user": "me", "text": "No rush, but hoping to merge by EOD if possible — it unblocks the Input component work.", "timestamp": "2026-05-04T16:16:00Z", "is_read": False},
]

_SEED_READ_STATES = [
    {"channel_id": "C001", "unread_count": 2},
    {"channel_id": "C002", "unread_count": 1},
    {"channel_id": "C003", "unread_count": 3},
    {"channel_id": "C004", "unread_count": 4},
    {"channel_id": "C005", "unread_count": 0},
    {"channel_id": "C006", "unread_count": 2},
    {"channel_id": "C007", "unread_count": 3},
]


_SEED_THREAD_REPLIES = [
    # m001 (alice all-hands reminder)
    {"id": "m001r1", "channel_id": "C001", "thread_parent_id": "m001", "user_name": "bob", "text": "Will the recording be posted afterward for folks in different timezones? 🌏", "timestamp": "2026-05-04T08:10:00Z", "is_pinned": False},
    {"id": "m001r2", "channel_id": "C001", "thread_parent_id": "m001", "user_name": "alice", "text": "Yes! Recording goes out within 24 hours via the company digest email. Also uploading to the shared drive for anyone who misses it.", "timestamp": "2026-05-04T08:15:00Z", "is_pinned": False},

    # m002 (carol Q2 results)
    {"id": "m002r1", "channel_id": "C001", "thread_parent_id": "m002", "user_name": "alice", "text": "118%!! The sales team absolutely crushed it this quarter. Best result since our Series A 🏆", "timestamp": "2026-05-04T09:35:00Z", "is_pinned": False},
    {"id": "m002r2", "channel_id": "C001", "thread_parent_id": "m002", "user_name": "bob", "text": "That number is going on a poster in my home office. Can't believe we actually hit it.", "timestamp": "2026-05-04T09:40:00Z", "is_pinned": False},
    {"id": "m002r3", "channel_id": "C001", "thread_parent_id": "m002", "user_name": "dave", "text": "Hats off to everyone. Product, engineering, sales — each team played their part in getting here.", "timestamp": "2026-05-04T09:45:00Z", "is_pinned": False},
    {"id": "m002r4", "channel_id": "C001", "thread_parent_id": "m002", "user_name": "me", "text": "Let's keep this momentum going in Q3! Huge milestone for the company 🚀", "timestamp": "2026-05-04T09:50:00Z", "is_pinned": False},

    # m003 (eve onboarding docs)
    {"id": "m003r1", "channel_id": "C001", "thread_parent_id": "m003", "user_name": "alice", "text": "This is fantastic Eve! The step-by-step walkthrough is exactly what new hires have been asking for. Sharing with the HR team now.", "timestamp": "2026-05-03T17:00:00Z", "is_pinned": False},
    {"id": "m003r2", "channel_id": "C001", "thread_parent_id": "m003", "user_name": "bob", "text": "Found a small typo on the environment setup page — 'enviornment' should be 'environment'. Happy to fix if you share edit access!", "timestamp": "2026-05-03T17:30:00Z", "is_pinned": False},

    # m004 (dave deployment)
    {"id": "m004r1", "channel_id": "C001", "thread_parent_id": "m004", "user_name": "alice", "text": "Noted — I'll warn the customer success team so they don't file tickets during the window.", "timestamp": "2026-05-04T14:05:00Z", "is_pinned": False},
    {"id": "m004r2", "channel_id": "C001", "thread_parent_id": "m004", "user_name": "carol", "text": "Can you post an update here when it's back up? I'm monitoring a customer demo env that depends on the metrics page.", "timestamp": "2026-05-04T14:08:00Z", "is_pinned": False},
    {"id": "m004r3", "channel_id": "C001", "thread_parent_id": "m004", "user_name": "me", "text": "👍 Will keep an eye on error rates in Grafana during the window. Ping me if you need backup.", "timestamp": "2026-05-04T14:10:00Z", "is_pinned": False},

    # m006 (alice Northstar deal)
    {"id": "m006r1", "channel_id": "C001", "thread_parent_id": "m006", "user_name": "bob", "text": "This is HUGE. Biggest deal in company history? The whole team has been working toward this for months 🚀🚀🚀", "timestamp": "2026-05-02T11:05:00Z", "is_pinned": False},
    {"id": "m006r2", "channel_id": "C001", "thread_parent_id": "m006", "user_name": "carol", "text": "So exciting! Will this affect Q3 roadmap priorities? I imagine they'll want some enterprise features fast-tracked.", "timestamp": "2026-05-02T11:10:00Z", "is_pinned": False},
    {"id": "m006r3", "channel_id": "C001", "thread_parent_id": "m006", "user_name": "dave", "text": "Congrats to the sales team! What's the go-live timeline? Want to make sure infra is ready for their scale.", "timestamp": "2026-05-02T11:15:00Z", "is_pinned": False},
    {"id": "m006r4", "channel_id": "C001", "thread_parent_id": "m006", "user_name": "eve", "text": "We worked so hard to make this happen. The demo environment we built for them specifically really paid off!", "timestamp": "2026-05-02T11:20:00Z", "is_pinned": False},
    {"id": "m006r5", "channel_id": "C001", "thread_parent_id": "m006", "user_name": "me", "text": "Go-live is targeting July 1st — let's make sure the enterprise tier is rock solid before then. @alice can we loop Sarah in on the onboarding planning?", "timestamp": "2026-05-02T11:30:00Z", "is_pinned": False},

    # m010 (eve PR #312)
    {"id": "m010r1", "channel_id": "C002", "thread_parent_id": "m010", "user_name": "alice", "text": "Looking at it now — love the session pooling approach. One request: can we add an integration test for the rollback path? I want to make sure a failed migration doesn't leave the DB in a partial state.", "timestamp": "2026-05-04T10:15:00Z", "is_pinned": False},
    {"id": "m010r2", "channel_id": "C002", "thread_parent_id": "m010", "user_name": "bob", "text": "Seconding the rollback test. Also — line 247, the bare `except Exception` is too broad. Can you catch specific SQLAlchemy exceptions (OperationalError, IntegrityError) so we don't swallow unrelated bugs?", "timestamp": "2026-05-04T10:30:00Z", "is_pinned": False},
    {"id": "m010r3", "channel_id": "C002", "thread_parent_id": "m010", "user_name": "dave", "text": "Nit: default connection pool size (5) will be too small for prod. Worth making it configurable via env var so we can tune without a redeploy.", "timestamp": "2026-05-04T10:45:00Z", "is_pinned": False},
    {"id": "m010r4", "channel_id": "C002", "thread_parent_id": "m010", "user_name": "eve", "text": "All great feedback — pushed fixes: narrowed exception types, added rollback integration test, made pool size configurable via DATABASE_POOL_SIZE. Ready for re-review!", "timestamp": "2026-05-04T12:00:00Z", "is_pinned": False},

    # m011 (alice Python 3.12)
    {"id": "m011r1", "channel_id": "C002", "thread_parent_id": "m011", "user_name": "bob", "text": "Ran a quick dep audit — `cryptography` needs a bump to ≥41.0 and `pydantic` v1 is incompatible with 3.12. I can send a PR for both if useful.", "timestamp": "2026-05-04T11:15:00Z", "is_pinned": False},
    {"id": "m011r2", "channel_id": "C002", "thread_parent_id": "m011", "user_name": "alice", "text": "Please do! Let's target the sprint after v2.4 ships. I'll create a tracking Jira ticket.", "timestamp": "2026-05-04T11:30:00Z", "is_pinned": False},

    # m012 (bob flaky test)
    {"id": "m012r1", "channel_id": "C002", "thread_parent_id": "m012", "user_name": "carol", "text": "Nice detective work! Was this the test that was blocking the last two release candidates?", "timestamp": "2026-05-04T16:15:00Z", "is_pinned": False},
    {"id": "m012r2", "channel_id": "C002", "thread_parent_id": "m012", "user_name": "alice", "text": "YES. That test has been failing randomly for 6 weeks. I nearly gave up and skipped it in CI. So glad you found the root cause.", "timestamp": "2026-05-04T16:20:00Z", "is_pinned": False},
    {"id": "m012r3", "channel_id": "C002", "thread_parent_id": "m012", "user_name": "dave", "text": "We should add a note to the eng wiki about this class of race condition so we don't reintroduce it in the auth layer.", "timestamp": "2026-05-04T16:30:00Z", "is_pinned": False},

    # m013 (carol post-mortem)
    {"id": "m013r1", "channel_id": "C002", "thread_parent_id": "m013", "user_name": "me", "text": "Well written. One gap: the alerting section doesn't mention the 8-minute detection lag. Add an action item for lowering the PagerDuty threshold for DB connection pool saturation.", "timestamp": "2026-05-03T15:30:00Z", "is_pinned": False},
    {"id": "m013r2", "channel_id": "C002", "thread_parent_id": "m013", "user_name": "alice", "text": "Second that. Also want to confirm the runbook gets updated with manual recovery steps before we close the incident.", "timestamp": "2026-05-03T16:00:00Z", "is_pinned": False},

    # m014 (me async refactor)
    {"id": "m014r1", "channel_id": "C002", "thread_parent_id": "m014", "user_name": "alice", "text": "60% reduction is massive — already visible in Grafana. Great work on this one.", "timestamp": "2026-05-03T17:45:00Z", "is_pinned": False},
    {"id": "m014r2", "channel_id": "C002", "thread_parent_id": "m014", "user_name": "eve", "text": "Clean refactor too — the worker queue design is much easier to reason about than the old threading model.", "timestamp": "2026-05-03T18:00:00Z", "is_pinned": False},

    # m016 (bob ClickHouse)
    {"id": "m016r1", "channel_id": "C002", "thread_parent_id": "m016", "user_name": "alice", "text": "We used ClickHouse at my last company for analytics — incredible for time-series aggregations. Queries that took 4s in Postgres ran in 200ms. Key is understanding MergeTree partitioning.", "timestamp": "2026-05-01T14:30:00Z", "is_pinned": False},
    {"id": "m016r2", "channel_id": "C002", "thread_parent_id": "m016", "user_name": "carol", "text": "Main gotcha: JOINs are expensive. If your queries are mostly single-table aggregations it's a great fit. If you need lots of JOINs across tables, think carefully.", "timestamp": "2026-05-01T14:45:00Z", "is_pinned": False},
    {"id": "m016r3", "channel_id": "C002", "thread_parent_id": "m016", "user_name": "bob", "text": "That's my concern — our funnel analysis does a lot of per-session JOINs between events and sessions tables. Maybe CH isn't the right fit for that use case.", "timestamp": "2026-05-01T15:00:00Z", "is_pinned": False},
    {"id": "m016r4", "channel_id": "C002", "thread_parent_id": "m016", "user_name": "dave", "text": "Also worth looking at Tinybird — managed ClickHouse with a clean REST API. Removes the operational overhead significantly.", "timestamp": "2026-05-01T15:15:00Z", "is_pinned": False},
    {"id": "m016r5", "channel_id": "C002", "thread_parent_id": "m016", "user_name": "me", "text": "I'd benchmark on your actual query patterns before committing — synthetic benchmarks are misleading for ClickHouse. Happy to help set up the comparison.", "timestamp": "2026-05-01T15:30:00Z", "is_pinned": False},

    # m017 (alice new hire)
    {"id": "m017r1", "channel_id": "C002", "thread_parent_id": "m017", "user_name": "bob", "text": "So excited to have a dedicated VP Eng! Welcome aboard Sarah 🎉", "timestamp": "2026-05-04T11:05:00Z", "is_pinned": False},
    {"id": "m017r2", "channel_id": "C002", "thread_parent_id": "m017", "user_name": "carol", "text": "Can't wait to meet her. Google infra background is going to be so relevant for our scaling challenges.", "timestamp": "2026-05-04T11:10:00Z", "is_pinned": False},

    # m020 (carol Q3 roadmap)
    {"id": "m020r1", "channel_id": "C003", "thread_parent_id": "m020", "user_name": "alice", "text": "Really excited about AI-powered search — it's been on the wishlist forever. What's the projected timeline? End of Q3?", "timestamp": "2026-05-04T10:45:00Z", "is_pinned": False},
    {"id": "m020r2", "channel_id": "C003", "thread_parent_id": "m020", "user_name": "bob", "text": "Bulk actions is hands-down the #1 request from our top enterprise accounts. So glad to see it finally on the roadmap.", "timestamp": "2026-05-04T10:50:00Z", "is_pinned": False},
    {"id": "m020r3", "channel_id": "C003", "thread_parent_id": "m020", "user_name": "dave", "text": "For the new onboarding flow, I've been sketching ideas. Will drop mockups in #design by end of week.", "timestamp": "2026-05-04T10:55:00Z", "is_pinned": False},
    {"id": "m020r4", "channel_id": "C003", "thread_parent_id": "m020", "user_name": "carol", "text": "AI search is targeting end of Q3 — partnering with infra on the embedding pipeline. Will have a more detailed timeline next week.", "timestamp": "2026-05-04T11:00:00Z", "is_pinned": False},
    {"id": "m020r5", "channel_id": "C003", "thread_parent_id": "m020", "user_name": "eve", "text": "What about CSV export? Multiple enterprise customers have been asking. Is it on the roadmap or still in the backlog?", "timestamp": "2026-05-04T11:05:00Z", "is_pinned": False},
    {"id": "m020r6", "channel_id": "C003", "thread_parent_id": "m020", "user_name": "me", "text": "Export is Q4 based on current priorities, but could move if more enterprise accounts escalate. @carol should I flag the accounts asking for it?", "timestamp": "2026-05-04T11:10:00Z", "is_pinned": False},

    # m021 (dave Figma mockups)
    {"id": "m021r1", "channel_id": "C003", "thread_parent_id": "m021", "user_name": "carol", "text": "Really like Option B — the top-nav with improved cards feels familiar for enterprise users and reduces learning curve.", "timestamp": "2026-05-04T13:15:00Z", "is_pinned": False},
    {"id": "m021r2", "channel_id": "C003", "thread_parent_id": "m021", "user_name": "alice", "text": "Agree with Carol. Option A is visually bold but might confuse existing users. Option B is the safer bet.", "timestamp": "2026-05-04T13:20:00Z", "is_pinned": False},
    {"id": "m021r3", "channel_id": "C003", "thread_parent_id": "m021", "user_name": "me", "text": "Could we do a hybrid? Option B's nav structure but Option C's metrics panel layout — it showed more data at a glance.", "timestamp": "2026-05-04T13:30:00Z", "is_pinned": False},
    {"id": "m021r4", "channel_id": "C003", "thread_parent_id": "m021", "user_name": "dave", "text": "B + C hybrid is a great idea! Will update Figma by EOD. Should get the best of both options.", "timestamp": "2026-05-04T13:45:00Z", "is_pinned": False},

    # m022 (me user research)
    {"id": "m022r1", "channel_id": "C003", "thread_parent_id": "m022", "user_name": "carol", "text": "Really valuable! Can you share the full notes doc? Want to use the specific pain points to inform roadmap prioritization.", "timestamp": "2026-05-03T11:15:00Z", "is_pinned": False},
    {"id": "m022r2", "channel_id": "C003", "thread_parent_id": "m022", "user_name": "alice", "text": "The episode reset speed is something we've heard from multiple accounts. Is there a Jira ticket tracking the investigation? I want to link it to our enterprise success notes.", "timestamp": "2026-05-03T11:30:00Z", "is_pinned": False},

    # m023 (alice NPS 72)
    {"id": "m023r1", "channel_id": "C003", "thread_parent_id": "m023", "user_name": "bob", "text": "72 is genuinely excellent — industry average for developer tools is around 30-40. We're in exceptional territory.", "timestamp": "2026-05-02T16:05:00Z", "is_pinned": False},
    {"id": "m023r2", "channel_id": "C003", "thread_parent_id": "m023", "user_name": "carol", "text": "Love seeing snapshot/restore called out specifically in the responses. It's such a core part of the experience.", "timestamp": "2026-05-02T16:10:00Z", "is_pinned": False},
    {"id": "m023r3", "channel_id": "C003", "thread_parent_id": "m023", "user_name": "me", "text": "A jump of 11 points in a single month is really meaningful. Great validation that we're building the right things.", "timestamp": "2026-05-02T16:15:00Z", "is_pinned": False},

    # m024 (carol agent observation format)
    {"id": "m024r1", "channel_id": "C003", "thread_parent_id": "m024", "user_name": "me", "text": "It's in `docs/api/agent-protocol.md` in the main repo — Section 3 covers the observation format with examples. Let me know if anything is unclear and I can add annotated samples.", "timestamp": "2026-05-01T14:00:00Z", "is_pinned": False},

    # m030 (bob tabs vs spaces)
    {"id": "m030r1", "channel_id": "C004", "thread_parent_id": "m030", "user_name": "alice", "text": "I am not having this argument again 😂 spaces, obviously. Always spaces.", "timestamp": "2026-05-04T12:05:00Z", "is_pinned": False},
    {"id": "m030r2", "channel_id": "C004", "thread_parent_id": "m030", "user_name": "carol", "text": "Spaces for consistent rendering across every editor, terminal, and GitHub PR diff. It's not even close.", "timestamp": "2026-05-04T12:10:00Z", "is_pinned": False},
    {"id": "m030r3", "channel_id": "C004", "thread_parent_id": "m030", "user_name": "dave", "text": "The correct answer: use an autoformatter, configure it in the linter, and never let humans make this decision again. Both sides are wrong for having opinions.", "timestamp": "2026-05-04T12:15:00Z", "is_pinned": False},
    {"id": "m030r4", "channel_id": "C004", "thread_parent_id": "m030", "user_name": "eve", "text": "I once worked at a company that used 3-space indentation. No one could explain why. I still see it in my dreams.", "timestamp": "2026-05-04T12:20:00Z", "is_pinned": False},
    {"id": "m030r5", "channel_id": "C004", "thread_parent_id": "m030", "user_name": "me", "text": "Tabs are semantically correct — they represent indentation, a logical concept. Spaces represent visual alignment, a presentation concern. Tabs are objectively right. Fight me.", "timestamp": "2026-05-04T12:25:00Z", "is_pinned": False},
    {"id": "m030r6", "channel_id": "C004", "thread_parent_id": "m030", "user_name": "alice", "text": "Me and Bob vs the entire company. This is how teams fracture. Filing this under 'unnecessary hill to die on' 😂", "timestamp": "2026-05-04T12:30:00Z", "is_pinned": False},
    {"id": "m030r7", "channel_id": "C004", "thread_parent_id": "m030", "user_name": "bob", "text": "At least @me and I are on the right side of history. We are the enlightened ones. The rest of you will understand eventually.", "timestamp": "2026-05-04T12:35:00Z", "is_pinned": False},

    # m031 (carol playoffs watch party)
    {"id": "m031r1", "channel_id": "C004", "thread_parent_id": "m031", "user_name": "bob", "text": "Absolutely in! Is it the Celtics game tonight?", "timestamp": "2026-05-04T13:05:00Z", "is_pinned": False},
    {"id": "m031r2", "channel_id": "C004", "thread_parent_id": "m031", "user_name": "dave", "text": "Game 5 at 8pm. I'm bringing chips and guac if we do the breakroom thing 🏀", "timestamp": "2026-05-04T13:10:00Z", "is_pinned": False},
    {"id": "m031r3", "channel_id": "C004", "thread_parent_id": "m031", "user_name": "alice", "text": "I have a customer call at 8pm unfortunately. Please send the score at halftime!", "timestamp": "2026-05-04T13:15:00Z", "is_pinned": False},
    {"id": "m031r4", "channel_id": "C004", "thread_parent_id": "m031", "user_name": "me", "text": "I'll be there! This is the most important game in years 🏀🔥", "timestamp": "2026-05-04T13:20:00Z", "is_pinned": False},

    # m032 (dave Claude chess)
    {"id": "m032r1", "channel_id": "C004", "thread_parent_id": "m032", "user_name": "bob", "text": "Wait what. Does it actually play legal moves? Does it know the piece rules?", "timestamp": "2026-05-04T14:35:00Z", "is_pinned": False},
    {"id": "m032r2", "channel_id": "C004", "thread_parent_id": "m032", "user_name": "carol", "text": "I tried it — opens with Sicilian Defense, plays real openings, and sacrifices pawns for positional pressure. Genuinely impressive chess bot.", "timestamp": "2026-05-04T14:40:00Z", "is_pinned": False},
    {"id": "m032r3", "channel_id": "C004", "thread_parent_id": "m032", "user_name": "dave", "text": "It blundered its queen in the endgame and then described it as 'accepting the sacrifice strategically'. 8/10 chess partner.", "timestamp": "2026-05-04T14:45:00Z", "is_pinned": False},

    # m033 (eve tacos for sprint goal)
    {"id": "m033r1", "channel_id": "C004", "thread_parent_id": "m033", "user_name": "alice", "text": "I'm IN. We deserve this and more. I believe in us 🌮", "timestamp": "2026-05-03T17:05:00Z", "is_pinned": False},
    {"id": "m033r2", "channel_id": "C004", "thread_parent_id": "m033", "user_name": "bob", "text": "Those 3 story points are now the only thing standing between me and tacos. Nothing has ever motivated me more.", "timestamp": "2026-05-03T17:10:00Z", "is_pinned": False},
    {"id": "m033r3", "channel_id": "C004", "thread_parent_id": "m033", "user_name": "carol", "text": "Please. Yes. I've been thinking about that taco place since last time. What time?", "timestamp": "2026-05-03T17:15:00Z", "is_pinned": False},
    {"id": "m033r4", "channel_id": "C004", "thread_parent_id": "m033", "user_name": "dave", "text": "12:30pm? I have a 1:30 but can make it work if we leave on time.", "timestamp": "2026-05-03T17:20:00Z", "is_pinned": False},
    {"id": "m033r5", "channel_id": "C004", "thread_parent_id": "m033", "user_name": "me", "text": "12:30 works! Already blocked it in my calendar. See you all there 🌮", "timestamp": "2026-05-03T17:25:00Z", "is_pinned": False},

    # m050 (dave design system kickoff)
    {"id": "m050r1", "channel_id": "C006", "thread_parent_id": "m050", "user_name": "carol", "text": "Excited about this! Consistent components will save so much time across projects. What's the first component we're tackling?", "timestamp": "2026-05-04T10:40:00Z", "is_pinned": False},
    {"id": "m050r2", "channel_id": "C006", "thread_parent_id": "m050", "user_name": "eve", "text": "I'd vote Button and Input first — they're the highest-frequency components and getting them right sets the tone for everything else.", "timestamp": "2026-05-04T10:45:00Z", "is_pinned": False},
    {"id": "m050r3", "channel_id": "C006", "thread_parent_id": "m050", "user_name": "dave", "text": "Starting with Button, Input, and Modal this sprint — then Card. Figma master file with all variants is now linked in the channel description.", "timestamp": "2026-05-04T10:50:00Z", "is_pinned": False},

    # m051 (carol component feedback)
    {"id": "m051r1", "channel_id": "C006", "thread_parent_id": "m051", "user_name": "dave", "text": "Great catches! Updating hover states to use a consistent 8% shade delta across all variants. And yes — embarrassing to miss the focus ring. Adding it now.", "timestamp": "2026-05-04T11:45:00Z", "is_pinned": False},
    {"id": "m051r2", "channel_id": "C006", "thread_parent_id": "m051", "user_name": "carol", "text": "Updated Figma looks great! The consistency really shows. Ready to hand off to engineering?", "timestamp": "2026-05-04T12:00:00Z", "is_pinned": False},

    # m052 (eve mobile responsive issues)
    {"id": "m052r1", "channel_id": "C006", "thread_parent_id": "m052", "user_name": "dave", "text": "The sidebar overlap has been in the backlog for two sprints without getting prioritized. Want to take a first pass at the CSS fix? I can review the PR same day.", "timestamp": "2026-05-04T13:30:00Z", "is_pinned": False},
    {"id": "m052r2", "channel_id": "C006", "thread_parent_id": "m052", "user_name": "eve", "text": "On it! Starting with the sidebar and data table since those are the most visible issues. PR by end of week.", "timestamp": "2026-05-04T13:35:00Z", "is_pinned": False},

    # m053 (me color palette)
    {"id": "m053r1", "channel_id": "C006", "thread_parent_id": "m053", "user_name": "carol", "text": "Love the warmer primary blue — feels much more on-brand. And the fact that all new colors pass WCAG AA is really appreciated. Approving this direction!", "timestamp": "2026-05-04T14:45:00Z", "is_pinned": False},

    # m060 (alice AWS cost spike)
    {"id": "m060r1", "channel_id": "C007", "thread_parent_id": "m060", "user_name": "bob", "text": "Looking at CloudWatch now — elevated Lambda invocations from the evaluation pipeline. The new parallel evaluation setup is fanning out to significantly more functions than expected. This looks like the culprit.", "timestamp": "2026-05-04T10:15:00Z", "is_pinned": False},
    {"id": "m060r2", "channel_id": "C007", "thread_parent_id": "m060", "user_name": "me", "text": "I see it — the parallel evaluation introduced an unbounded fan-out. Adding a concurrency cap now. Pushing a fix shortly.", "timestamp": "2026-05-04T10:20:00Z", "is_pinned": False},
    {"id": "m060r3", "channel_id": "C007", "thread_parent_id": "m060", "user_name": "carol", "text": "How much are we over? Are we at risk of hitting the monthly budget alarm?", "timestamp": "2026-05-04T10:25:00Z", "is_pinned": False},
    {"id": "m060r4", "channel_id": "C007", "thread_parent_id": "m060", "user_name": "alice", "text": "78% of monthly cap with 12 days left. Not critical yet but need the concurrency fix in before the weekend eval jobs run.", "timestamp": "2026-05-04T10:30:00Z", "is_pinned": False},

    # m061 (bob k8s upgrade)
    {"id": "m061r1", "channel_id": "C007", "thread_parent_id": "m061", "user_name": "alice", "text": "Thanks for the heads up! What's the rollback plan if something breaks post-upgrade?", "timestamp": "2026-05-04T11:05:00Z", "is_pinned": False},
    {"id": "m061r2", "channel_id": "C007", "thread_parent_id": "m061", "user_name": "dave", "text": "1.27 → 1.28 is usually smooth. Main risk: deprecated API removals. Make sure workload manifests are updated for the new API versions before you start.", "timestamp": "2026-05-04T11:10:00Z", "is_pinned": False},
    {"id": "m061r3", "channel_id": "C007", "thread_parent_id": "m061", "user_name": "bob", "text": "Good call @dave — running `kubectl deprecations` scan now. Rollback plan: full etcd snapshot from last night + cluster state backup. Should be clean recovery if anything goes sideways.", "timestamp": "2026-05-04T11:15:00Z", "is_pinned": False},

    # m062 (me monitoring dashboard)
    {"id": "m062r1", "channel_id": "C007", "thread_parent_id": "m062", "user_name": "alice", "text": "This is exactly what oncall needed! Can you add a panel for active agent sessions? That's the metric I always want to see first during incidents.", "timestamp": "2026-05-04T12:15:00Z", "is_pinned": False},
    {"id": "m062r2", "channel_id": "C007", "thread_parent_id": "m062", "user_name": "me", "text": "Added the active sessions panel — live now. Also added p99 latency for the action API since it was requested in the last retro.", "timestamp": "2026-05-04T12:30:00Z", "is_pinned": False},
]


_CHANNEL_AUTO_RESPONDERS = {
    "general": [
        ("bob", "👍 noted!"),
        ("alice", "Thanks for the update!"),
    ],
    "engineering": [
        ("alice", "Thanks! I'll take a look."),
        ("carol", "Looks good to me 👀"),
    ],
    "product": [
        ("carol", "Good point, let's discuss in the next sync."),
        ("dave", "Agreed! Adding to the roadmap notes."),
    ],
    "random": [
        ("bob", "😂"),
        ("carol", "haha same!"),
    ],
    "design": [
        ("dave", "Nice! I'll review in Figma."),
        ("carol", "Looks great — leaving comments."),
    ],
    "ops-infra": [
        ("bob", "On it — checking CloudWatch now."),
        ("alice", "Thanks for flagging."),
    ],
}


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
        for r in _SEED_THREAD_REPLIES:
            db.add(Message(
                id=r["id"], channel_id=r["channel_id"], user_name=r["user_name"],
                text=r["text"], timestamp=r["timestamp"], is_pinned=r.get("is_pinned", False),
                thread_parent_id=r["thread_parent_id"], reply_count=0,
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
    # Auto-save baseline snapshot after seeding
    with SessionLocal() as snap_db:
        data = _dump_full_db(snap_db)
        snap_db.add(SavedState(slot="baseline", data=json.dumps(data)))
        snap_db.commit()


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
        db.query(ActionLog).delete()
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


class ReceiveDmRequest(BaseModel):
    from_user: str
    text: str


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _get_channel(db: Session, channel_ref: str) -> Channel | None:
    return db.query(Channel).filter(
        (Channel.id == channel_ref) | (Channel.name == channel_ref)
    ).first()


@app.post("/receive_dm")
def receive_dm(req: ReceiveDmRequest):
    """Inject an incoming DM. Used by evaluators."""
    with SessionLocal() as db:
        dm_id = f"dm{uuid.uuid4().hex[:8]}"
        db.add(DirectMessage(
            id=dm_id, from_user=req.from_user, to_user="me",
            text=req.text, timestamp=_now(), is_read=False,
        ))
        _log_action(db, "receive_dm", dm_id, {"from": req.from_user})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "received", "dm_id": dm_id, "state": state}


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
        _log_action(db, "send_message", msg_id, {"channel": channel.name, "text": req.text[:80]})
        db.commit()
        # Auto-inject channel responder if configured and not triggered in the last minute
        channel_name = channel.name
        if channel_name in _CHANNEL_AUTO_RESPONDERS:
            from datetime import timedelta
            one_minute_ago = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
            already = db.query(ActionLog).filter(
                ActionLog.action_type == "auto_response",
                ActionLog.payload.contains(f'"channel": "{channel_name}"'),
                ActionLog.timestamp >= one_minute_ago,
            ).first()
            if not already:
                responder_user, responder_text = _CHANNEL_AUTO_RESPONDERS[channel_name][0]
                resp_id = f"m{uuid.uuid4().hex[:8]}"
                db.add(Message(
                    id=resp_id, channel_id=channel.id, user_name=responder_user,
                    text=responder_text, timestamp=_now(), is_pinned=False,
                    thread_parent_id=None, reply_count=0,
                ))
                _log_action(db, "auto_response", resp_id, {"channel": channel_name, "from": responder_user})
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
        _log_action(db, "reply_thread", reply_id, {"message_id": req.message_id})
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
            _log_action(db, "add_reaction", req.message_id, {"emoji": req.emoji})
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
            _log_action(db, "remove_reaction", req.message_id, {"emoji": req.emoji})
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
        _log_action(db, "pin_message", req.message_id)
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
        _log_action(db, "unpin_message", req.message_id)
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
        was_pinned = bool(msg.is_pinned)
        db.query(Reaction).filter(Reaction.message_id == req.message_id).delete()
        db.delete(msg)
        _log_action(db, "delete_message", req.message_id, {"was_pinned": was_pinned})
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
        _log_action(db, "create_channel", channel_id, {"name": req.name})
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
        _log_action(db, "archive_channel", channel.id, {"name": channel.name})
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
        _log_action(db, "set_status", None, {"status": req.status, "emoji": req.emoji or ""})
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
        _log_action(db, "send_dm", dm_id, {"to": req.to})
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
        _log_action(db, "mark_channel_read", channel.id, {"channel": channel.name})
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
