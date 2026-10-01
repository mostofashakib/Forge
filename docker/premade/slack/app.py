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

from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import (
    Boolean, Column, Integer, String, Text, create_engine, func
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from forge_protocol import (
    BASELINE_SLOT, TIMESTAMP_FORMAT, Ledger, dump_ledger, forge_router, protocol_tables, restore_ledger, save_snapshot,
)
from slack_seed import CHANNEL_AUTO_RESPONDERS, SEED_CHANNELS, SEED_DMS, SEED_MESSAGES, SEED_REACTIONS, SEED_READ_STATES, SEED_THREAD_REPLIES

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

    id = Column(String, primary_key=True)
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


_TABLES = protocol_tables(Base)
ActionLog = _TABLES.action_log

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

def _counts_by_channel(db: Session, *criteria) -> dict[str, int]:
    return dict(
        db.query(Message.channel_id, func.count())
        .filter(*criteria)
        .group_by(Message.channel_id)
        .all()
    )


def _channel_dicts(channels: list[Channel], db: Session) -> list[dict]:
    """Channel summaries from three grouped queries, however many channels exist."""
    message_counts = _counts_by_channel(db, Message.thread_parent_id == None)  # noqa: E711
    pinned_counts = _counts_by_channel(db, Message.is_pinned == True)  # noqa: E712
    unread = dict(db.query(ChannelReadState.channel_id, ChannelReadState.unread_count).all())
    return [
        {
            "id": c.id,
            "name": c.name,
            "purpose": c.purpose,
            "is_private": c.is_private,
            "archived": c.is_archived,
            "message_count": message_counts.get(c.id, 0),
            "pinned_count": pinned_counts.get(c.id, 0),
            "unread": unread.get(c.id, 0),
        }
        for c in channels
    ]


def _message_dicts(messages: list[Message], db: Session) -> list[dict]:
    """Messages with their reactions, loaded in one query."""
    reactions: dict[str, dict[str, list[str]]] = {m.id: {} for m in messages}
    if messages:
        for r in db.query(Reaction).filter(Reaction.message_id.in_(list(reactions))).all():
            reactions[r.message_id].setdefault(r.emoji, []).append(r.user_name)
    return [_message_to_dict(m, reactions[m.id]) for m in messages]


def _message_to_dict(m: Message, reactions: dict[str, list[str]]) -> dict:
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
    channel_dicts = _channel_dicts(channels, db)
    total_unread = dm_unread + sum(c["unread"] for c in channel_dicts)
    return {
        "workspace_name": "Acme Corp",
        "channels": channel_dicts,
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
    return {
        "channels": channels, "messages": messages, "reactions": reactions,
        "direct_messages": dms, "user_statuses": statuses, "read_states": read_states,
        **dump_ledger(db, _TABLES),
    }


def _restore_from_dict(db: Session, data: dict) -> None:
    db.query(Reaction).delete()
    db.query(Message).delete()
    db.query(Channel).delete()
    db.query(DirectMessage).delete()
    db.query(UserStatus).delete()
    db.query(ChannelReadState).delete()
    restore_ledger(db, _TABLES, data)
    for c in data.get("channels", []):
        db.add(Channel(
            id=c["id"], name=c["name"], purpose=c.get("purpose", ""),
            is_private=c.get("is_private", False), is_archived=c.get("is_archived", c.get("archived", False)),
            created_at=c["created_at"] if "created_at" in c else _ledger.now(db),
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


# Virtual time starts a minute after the newest seed message, so anything
# created during an episode sorts as the newest activity.
_ledger = Ledger(_TABLES, epoch=max(
    datetime.strptime(item["timestamp"], TIMESTAMP_FORMAT)
    for item in (*SEED_MESSAGES, *SEED_THREAD_REPLIES, *SEED_DMS)
) + timedelta(minutes=1))


def _seed_if_empty() -> None:
    with SessionLocal() as db:
        if db.query(Channel).count() > 0:
            return
        for c in SEED_CHANNELS:
            db.add(Channel(
                id=c["id"], name=c["name"], purpose=c["purpose"],
                is_private=c["is_private"], is_archived=c["is_archived"],
                created_at="2026-01-01T00:00:00Z",
            ))
        for m in SEED_MESSAGES:
            db.add(Message(
                id=m["id"], channel_id=m["channel_id"], user_name=m["user_name"],
                text=m["text"], timestamp=m["timestamp"], is_pinned=m["is_pinned"],
                thread_parent_id=None, reply_count=m["reply_count"],
            ))
        for r in SEED_THREAD_REPLIES:
            db.add(Message(
                id=r["id"], channel_id=r["channel_id"], user_name=r["user_name"],
                text=r["text"], timestamp=r["timestamp"], is_pinned=r.get("is_pinned", False),
                thread_parent_id=r["thread_parent_id"], reply_count=0,
            ))
        for r in SEED_REACTIONS:
            db.add(Reaction(id=r["id"], message_id=r["message_id"], emoji=r["emoji"], user_name=r["user_name"]))
        for dm in SEED_DMS:
            db.add(DirectMessage(
                id=dm["id"], from_user=dm["from_user"], to_user=dm["to_user"],
                text=dm["text"], timestamp=dm["timestamp"], is_read=dm["is_read"],
            ))
        db.add(UserStatus(user_name="me", status_text="", status_emoji="", presence="online"))
        for rs in SEED_READ_STATES:
            db.add(ChannelReadState(channel_id=rs["channel_id"], unread_count=rs["unread_count"]))
        db.commit()
    with SessionLocal() as db:
        save_snapshot(db, _TABLES, BASELINE_SLOT, _dump_full_db(db))


_seed_if_empty()

app.include_router(forge_router(
    SessionLocal, _TABLES,
    state=_get_state_dict, dump=_dump_full_db, restore=_restore_from_dict,
    domain_tables=(Reaction, Message, Channel, DirectMessage, UserStatus, ChannelReadState),
    seed=_seed_if_empty,
))


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
        dm_id = _ledger.next_id(db, "dm")
        db.add(DirectMessage(
            id=dm_id, from_user=req.from_user, to_user="me",
            text=req.text, timestamp=_ledger.now(db), is_read=False,
        ))
        _ledger.log_action(db, "receive_dm", dm_id, {"from": req.from_user})
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
        msg_id = _ledger.next_id(db, "m")
        db.add(Message(
            id=msg_id, channel_id=channel.id, user_name="me",
            text=req.text, timestamp=_ledger.now(db), is_pinned=False,
            thread_parent_id=None, reply_count=0,
        ))
        _ledger.log_action(db, "send_message", msg_id, {"channel": channel.name, "text": req.text[:80]})
        db.commit()
        # Auto-inject channel responder if configured and not triggered in the last minute
        channel_name = channel.name
        if channel_name in CHANNEL_AUTO_RESPONDERS:
            one_minute_ago = _ledger.at(_ledger.elapsed(db) - 60)
            already = db.query(ActionLog).filter(
                ActionLog.action_type == "auto_response",
                ActionLog.payload.contains(f'"channel": "{channel_name}"'),
                ActionLog.timestamp >= one_minute_ago,
            ).first()
            if not already:
                responder_user, responder_text = CHANNEL_AUTO_RESPONDERS[channel_name][0]
                resp_id = _ledger.next_id(db, "m")
                db.add(Message(
                    id=resp_id, channel_id=channel.id, user_name=responder_user,
                    text=responder_text, timestamp=_ledger.now(db), is_pinned=False,
                    thread_parent_id=None, reply_count=0,
                ))
                _ledger.log_action(db, "auto_response", resp_id, {"channel": channel_name, "from": responder_user})
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
        reply_id = _ledger.next_id(db, "t")
        db.add(Message(
            id=reply_id, channel_id=channel.id, user_name="me",
            text=req.text, timestamp=_ledger.now(db), is_pinned=False,
            thread_parent_id=req.message_id, reply_count=0,
        ))
        parent.reply_count += 1
        _ledger.log_action(db, "reply_thread", reply_id, {"message_id": req.message_id})
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
                id=_ledger.next_id(db, "rx"),
                message_id=req.message_id, emoji=req.emoji, user_name="me",
            ))
            _ledger.log_action(db, "add_reaction", req.message_id, {"emoji": req.emoji})
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
            _ledger.log_action(db, "remove_reaction", req.message_id, {"emoji": req.emoji})
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
        _ledger.log_action(db, "pin_message", req.message_id)
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
        _ledger.log_action(db, "unpin_message", req.message_id)
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
        _ledger.log_action(db, "delete_message", req.message_id, {"was_pinned": was_pinned})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "deleted", "message_id": req.message_id, "state": state}


@app.post("/create_channel")
def create_channel(req: CreateChannelRequest):
    with SessionLocal() as db:
        existing = _get_channel(db, req.name)
        if existing:
            return {"status": "error", "message": f"Channel '{req.name}' already exists", "state": _get_state_dict(db)}
        channel_id = _ledger.next_id(db, "C")
        db.add(Channel(
            id=channel_id,
            name=req.name.lower().replace(" ", "-"),
            purpose=req.purpose or "",
            is_private=False,
            is_archived=False,
            created_at=_ledger.now(db),
        ))
        db.add(ChannelReadState(channel_id=channel_id, unread_count=0))
        _ledger.log_action(db, "create_channel", channel_id, {"name": req.name})
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
        _ledger.log_action(db, "archive_channel", channel.id, {"name": channel.name})
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
        _ledger.log_action(db, "set_status", None, {"status": req.status, "emoji": req.emoji or ""})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "updated", "new_status": req.status, "emoji": req.emoji, "state": state}


@app.post("/send_dm")
def send_dm(req: SendDmRequest):
    with SessionLocal() as db:
        dm_id = _ledger.next_id(db, "dm")
        db.add(DirectMessage(
            id=dm_id, from_user="me", to_user=req.to,
            text=req.text, timestamp=_ledger.now(db), is_read=True,
        ))
        _ledger.log_action(db, "send_dm", dm_id, {"to": req.to})
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
        _ledger.log_action(db, "mark_channel_read", channel.id, {"channel": channel.name})
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
        results = _message_dicts(
            [m for m in messages if q in m.text.lower() or q in m.user_name.lower()], db
        )
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
        result = _message_dicts(messages, db)
    return {"channel": chan_ref, "messages": result, "count": len(result)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
