"""Gmail-like Forge RL environment — high-fidelity SQLAlchemy/SQLite version.

Forge protocol:
  GET  /forge/health
  GET  /forge/state
  POST /forge/reset
  POST /forge/snapshot        body: {"slot": "name"}
  POST /forge/restore/{slot}
  POST /forge/restore-state   body: full state JSON

Action endpoints (all POST):
  /compose  /send  /reply  /forward  /archive  /delete
  /mark_read  /star  /label  /search  /move
  /bulk_archive  /create_label  /empty_trash  /get_thread
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import (
    Boolean, Column, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from forge_protocol import (
    BASELINE_SLOT, TIMESTAMP_FORMAT, Ledger, dump_ledger, forge_router, protocol_tables, restore_ledger, save_snapshot,
)
from gmail_seed import AUTO_REPLY_MAP, SEED_CONTACTS, SEED_EMAILS, SEED_LABELS

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite:///./gmail.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Email(Base):
    __tablename__ = "emails"

    id = Column(String, primary_key=True)
    thread_id = Column(String, nullable=True)
    folder = Column(String, default="inbox")
    from_addr = Column(String, nullable=False)
    to_addr = Column(String, nullable=False)
    cc = Column(String, default="")
    subject = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    snippet = Column(String, default="")
    is_read = Column(Boolean, default=False)
    is_starred = Column(Boolean, default=False)
    labels = Column(Text, default="[]")  # JSON list
    timestamp = Column(String, nullable=False)
    has_attachment = Column(Boolean, default=False)


class Contact(Base):
    __tablename__ = "contacts"

    email = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    avatar_initial = Column(String, default="?")


class Label(Base):
    __tablename__ = "labels"

    id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    color = Column(String, default="#1a73e8")


_TABLES = protocol_tables(Base)
ActionLog = _TABLES.action_log

Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Gmail-like Environment", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snippet(body: str) -> str:
    return body[:100].replace("\n", " ").strip()


def _email_to_dict(e: Email) -> dict:
    return {
        "id": e.id,
        "thread_id": e.thread_id,
        "folder": e.folder,
        "from": e.from_addr,
        "to": e.to_addr,
        "cc": e.cc,
        "subject": e.subject,
        "body": e.body,
        "snippet": e.snippet or _snippet(e.body),
        "is_read": e.is_read,
        "is_starred": e.is_starred,
        "labels": json.loads(e.labels or "[]"),
        "timestamp": e.timestamp,
        "has_attachment": e.has_attachment,
    }


def _contact_to_dict(c: Contact) -> dict:
    return {"email": c.email, "name": c.name, "avatar_initial": c.avatar_initial}


def _label_to_dict(lb: Label) -> dict:
    return {"id": lb.id, "name": lb.name, "color": lb.color}


def _get_state_dict(db: Session) -> dict:
    all_emails = db.query(Email).all()
    inbox = [_email_to_dict(e) for e in all_emails if e.folder == "inbox"]
    inbox_sorted = sorted(inbox, key=lambda x: x["timestamp"], reverse=True)
    labels = [_label_to_dict(lb) for lb in db.query(Label).all()]
    contacts = [_contact_to_dict(c) for c in db.query(Contact).all()]
    return {
        "inbox": inbox_sorted,
        "inbox_unread": sum(1 for e in inbox if not e["is_read"]),
        "sent_count": sum(1 for e in all_emails if e.folder == "sent"),
        "draft_count": sum(1 for e in all_emails if e.folder == "drafts"),
        "trash_count": sum(1 for e in all_emails if e.folder == "trash"),
        "archive_count": sum(1 for e in all_emails if e.folder == "archive"),
        "labels": labels,
        "contacts": contacts,
        "total_emails": len(all_emails),
        "starred_emails": [_email_to_dict(e) for e in all_emails if e.is_starred],
        "inbox_count": sum(1 for e in all_emails if e.folder == "inbox"),
        "actions_taken": db.query(ActionLog).count(),
        "recent_actions": [
            {"id": a.id, "action_type": a.action_type, "target_id": a.target_id, "timestamp": a.timestamp}
            for a in db.query(ActionLog).order_by(ActionLog.timestamp.desc()).limit(10).all()
        ],
    }


def _dump_full_db(db: Session) -> dict:
    return {
        "emails": [_email_to_dict(e) for e in db.query(Email).all()],
        "contacts": [_contact_to_dict(c) for c in db.query(Contact).all()],
        "labels": [_label_to_dict(lb) for lb in db.query(Label).all()],
        **dump_ledger(db, _TABLES),
    }


def _restore_from_dict(db: Session, data: dict) -> None:
    db.query(Email).delete()
    db.query(Contact).delete()
    db.query(Label).delete()
    restore_ledger(db, _TABLES, data)
    for e in data.get("emails", []):
        db.add(Email(
            id=e["id"],
            thread_id=e.get("thread_id"),
            folder=e.get("folder", "inbox"),
            from_addr=e.get("from", e.get("from_addr", "")),
            to_addr=e.get("to", e.get("to_addr", "")),
            cc=e.get("cc", ""),
            subject=e.get("subject", ""),
            body=e.get("body", ""),
            snippet=e.get("snippet", ""),
            is_read=e.get("is_read", False),
            is_starred=e.get("is_starred", False),
            labels=json.dumps(e.get("labels", [])),
            timestamp=e["timestamp"] if "timestamp" in e else _ledger.now(db),
            has_attachment=e.get("has_attachment", False),
        ))
    for c in data.get("contacts", []):
        db.add(Contact(
            email=c["email"],
            name=c["name"],
            avatar_initial=c.get("avatar_initial", c["name"][0].upper()),
        ))
    for lb in data.get("labels", []):
        db.add(Label(id=lb["id"], name=lb["name"], color=lb.get("color", "#1a73e8")))
    db.commit()


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------


# Virtual time starts a minute after the newest seed email, so anything created
# during an episode sorts as the newest mail.
_ledger = Ledger(_TABLES, epoch=max(
    datetime.strptime(e["timestamp"], TIMESTAMP_FORMAT) for e in SEED_EMAILS
) + timedelta(minutes=1))


def _seed_if_empty() -> None:
    with SessionLocal() as db:
        if db.query(Email).count() > 0:
            return
        for e in SEED_EMAILS:
            snippet = _snippet(e["body"])
            db.add(Email(
                id=e["id"],
                thread_id=e.get("thread_id"),
                folder=e["folder"],
                from_addr=e["from_addr"],
                to_addr=e["to_addr"],
                cc=e.get("cc", ""),
                subject=e["subject"],
                body=e["body"],
                snippet=snippet,
                is_read=e.get("is_read", False),
                is_starred=e.get("is_starred", False),
                labels=e.get("labels", "[]"),
                timestamp=e["timestamp"],
                has_attachment=e.get("has_attachment", False),
            ))
        for c in SEED_CONTACTS:
            db.add(Contact(**c))
        for lb in SEED_LABELS:
            db.add(Label(**lb))
        db.commit()
    with SessionLocal() as db:
        save_snapshot(db, _TABLES, BASELINE_SLOT, _dump_full_db(db))


_seed_if_empty()

app.include_router(forge_router(
    SessionLocal, _TABLES,
    state=_get_state_dict, dump=_dump_full_db, restore=_restore_from_dict,
    domain_tables=(Email, Contact, Label), seed=_seed_if_empty,
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

class ComposeRequest(BaseModel):
    to: str
    subject: str
    body: str = ""
    cc: Optional[str] = ""


class SendRequest(BaseModel):
    draft_id: str


class ReplyRequest(BaseModel):
    email_id: str
    body: str


class ForwardRequest(BaseModel):
    email_id: str
    to: str
    note: Optional[str] = ""


class EmailIdRequest(BaseModel):
    email_id: str


class MarkReadRequest(BaseModel):
    email_id: str
    read: bool = True


class StarRequest(BaseModel):
    email_id: str
    starred: bool = True


class LabelRequest(BaseModel):
    email_id: str
    label: str
    add: bool = True


class SearchRequest(BaseModel):
    query: str
    folder: Optional[str] = None


class MoveRequest(BaseModel):
    email_id: str
    folder: str


class BulkArchiveRequest(BaseModel):
    email_ids: list[str]


class CreateLabelRequest(BaseModel):
    name: str
    color: str = "#1a73e8"


class GetThreadRequest(BaseModel):
    thread_id: str


class ReceiveRequest(BaseModel):
    from_addr: str
    subject: str
    body: str
    thread_id: Optional[str] = None
    cc: Optional[str] = ""
    labels: Optional[list[str]] = []


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@app.post("/receive")
def receive(req: ReceiveRequest):
    """Inject an incoming email into the inbox. Used by evaluators to simulate responses."""
    with SessionLocal() as db:
        email_id = _ledger.next_id(db, "i")
        db.add(Email(
            id=email_id,
            thread_id=req.thread_id,
            folder="inbox",
            from_addr=req.from_addr,
            to_addr="me@company.com",
            cc=req.cc or "",
            subject=req.subject,
            body=req.body,
            snippet=_snippet(req.body),
            is_read=False,
            is_starred=False,
            labels=json.dumps(req.labels or []),
            timestamp=_ledger.now(db),
            has_attachment=False,
        ))
        _ledger.log_action(db, "receive", email_id, {"from": req.from_addr, "subject": req.subject})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "received", "email_id": email_id, "state": state}


@app.post("/compose")
def compose(req: ComposeRequest):
    with SessionLocal() as db:
        draft_id = _ledger.next_id(db, "d")
        ts = _ledger.now(db)
        db.add(Email(
            id=draft_id,
            thread_id=None,
            folder="drafts",
            from_addr="me@company.com",
            to_addr=req.to,
            cc=req.cc or "",
            subject=req.subject,
            body=req.body,
            snippet=_snippet(req.body),
            is_read=True,
            is_starred=False,
            labels="[]",
            timestamp=ts,
            has_attachment=False,
        ))
        _ledger.log_action(db, "compose", draft_id)
        db.commit()
        state = _get_state_dict(db)
    return {"status": "draft_created", "draft_id": draft_id, "state": state}


@app.post("/send")
def send(req: SendRequest):
    with SessionLocal() as db:
        email = db.query(Email).filter(Email.id == req.draft_id).first()
        if not email or email.folder != "drafts":
            return {"status": "error", "message": f"Draft '{req.draft_id}' not found", "state": _get_state_dict(db)}
        to_addr = email.to_addr
        email.folder = "sent"
        email.timestamp = _ledger.now(db)
        _ledger.log_action(db, "send", req.draft_id, {"to": to_addr})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "sent", "email_id": req.draft_id, "state": state}


@app.post("/reply")
def reply(req: ReplyRequest):
    with SessionLocal() as db:
        original = db.query(Email).filter(Email.id == req.email_id).first()
        if not original:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        reply_id = _ledger.next_id(db, "r")
        orig_labels = original.labels or "[]"
        db.add(Email(
            id=reply_id,
            thread_id=original.thread_id or original.id,
            folder="sent",
            from_addr="me@company.com",
            to_addr=original.from_addr,
            cc="",
            subject=f"Re: {original.subject}" if not original.subject.startswith("Re:") else original.subject,
            body=req.body,
            snippet=_snippet(req.body),
            is_read=True,
            is_starred=False,
            labels=orig_labels,
            timestamp=_ledger.now(db),
            has_attachment=False,
        ))
        original.is_read = True
        thread = original.thread_id or original.id
        orig_from = original.from_addr
        _ledger.log_action(db, "reply", reply_id, {"to": orig_from})
        db.commit()
        # Auto-inject simulated response if configured and not already done
        if thread in AUTO_REPLY_MAP:
            already = db.query(ActionLog).filter(
                ActionLog.action_type == "auto_reply",
                ActionLog.target_id == thread,
            ).first()
            if not already:
                r = AUTO_REPLY_MAP[thread]
                resp_id = _ledger.next_id(db, "ar")
                db.add(Email(
                    id=resp_id, thread_id=thread, folder="inbox",
                    from_addr=r["from_addr"], to_addr="me@company.com", cc="",
                    subject=r["subject"], body=r["body"],
                    snippet=_snippet(r["body"]),
                    is_read=False, is_starred=False, labels='["work"]',
                    timestamp=_ledger.now(db), has_attachment=False,
                ))
                _ledger.log_action(db, "auto_reply", thread, {"from": r["from_addr"]})
                db.commit()
        state = _get_state_dict(db)
    return {"status": "replied", "reply_id": reply_id, "state": state}


@app.post("/forward")
def forward(req: ForwardRequest):
    with SessionLocal() as db:
        original = db.query(Email).filter(Email.id == req.email_id).first()
        if not original:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        fwd_id = _ledger.next_id(db, "f")
        note_text = f"{req.note}\n\n" if req.note else ""
        fwd_body = (
            f"{note_text}"
            f"---------- Forwarded message ----------\n"
            f"From: {original.from_addr}\n"
            f"Subject: {original.subject}\n\n"
            f"{original.body}"
        )
        db.add(Email(
            id=fwd_id,
            thread_id=None,
            folder="sent",
            from_addr="me@company.com",
            to_addr=req.to,
            cc="",
            subject=f"Fwd: {original.subject}",
            body=fwd_body,
            snippet=_snippet(fwd_body),
            is_read=True,
            is_starred=False,
            labels="[]",
            timestamp=_ledger.now(db),
            has_attachment=original.has_attachment,
        ))
        _ledger.log_action(db, "forward", fwd_id, {"to": req.to})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "forwarded", "forward_id": fwd_id, "state": state}


@app.post("/archive")
def archive(req: EmailIdRequest):
    with SessionLocal() as db:
        email = db.query(Email).filter(Email.id == req.email_id).first()
        if not email:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        email.folder = "archive"
        _ledger.log_action(db, "archive", req.email_id)
        db.commit()
        state = _get_state_dict(db)
    return {"status": "archived", "email_id": req.email_id, "state": state}


@app.post("/delete")
def delete(req: EmailIdRequest):
    with SessionLocal() as db:
        email = db.query(Email).filter(Email.id == req.email_id).first()
        if not email:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        was_starred = bool(email.is_starred)
        subject = email.subject
        email.folder = "trash"
        _ledger.log_action(db, "delete", req.email_id, {"subject": subject, "was_starred": was_starred})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "deleted", "email_id": req.email_id, "state": state}


@app.post("/mark_read")
def mark_read(req: MarkReadRequest):
    with SessionLocal() as db:
        email = db.query(Email).filter(Email.id == req.email_id).first()
        if not email:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        email.is_read = req.read
        _ledger.log_action(db, "mark_read", req.email_id, {"read": req.read})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "marked", "email_id": req.email_id, "read": req.read, "state": state}


@app.post("/star")
def star(req: StarRequest):
    with SessionLocal() as db:
        email = db.query(Email).filter(Email.id == req.email_id).first()
        if not email:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        email.is_starred = req.starred
        _ledger.log_action(db, "star", req.email_id, {"starred": req.starred})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "starred", "email_id": req.email_id, "starred": req.starred, "state": state}


@app.post("/label")
def label(req: LabelRequest):
    with SessionLocal() as db:
        email = db.query(Email).filter(Email.id == req.email_id).first()
        if not email:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        current = json.loads(email.labels or "[]")
        if req.add:
            if req.label not in current:
                current.append(req.label)
            if not db.query(Label).filter(Label.name == req.label).first():
                db.add(Label(id=_ledger.next_id(db, "l"), name=req.label, color="#1a73e8"))
        else:
            current = [l for l in current if l != req.label]
        email.labels = json.dumps(current)
        _ledger.log_action(db, "label", req.email_id)
        db.commit()
        state = _get_state_dict(db)
    return {"status": "labeled", "email_id": req.email_id, "label": req.label, "added": req.add, "state": state}


@app.post("/search")
def search(req: SearchRequest):
    q = req.query.lower()
    with SessionLocal() as db:
        query = db.query(Email)
        if req.folder:
            query = query.filter(Email.folder == req.folder)
        emails = query.all()
        results = [
            _email_to_dict(e) for e in emails
            if q in e.subject.lower()
            or q in e.body.lower()
            or q in e.from_addr.lower()
            or q in e.to_addr.lower()
        ]
        state = _get_state_dict(db)
    return {"results": results, "count": len(results), "state": state}


@app.post("/move")
def move(req: MoveRequest):
    valid_folders = {"inbox", "sent", "drafts", "trash", "archive"}
    if req.folder not in valid_folders:
        return {"status": "error", "message": f"Invalid folder '{req.folder}'"}
    with SessionLocal() as db:
        email = db.query(Email).filter(Email.id == req.email_id).first()
        if not email:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        email.folder = req.folder
        _ledger.log_action(db, "move", req.email_id, {"folder": req.folder})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "moved", "email_id": req.email_id, "folder": req.folder, "state": state}


@app.post("/bulk_archive")
def bulk_archive(req: BulkArchiveRequest):
    with SessionLocal() as db:
        found = {e.id: e for e in db.query(Email).filter(Email.id.in_(req.email_ids)).all()}
        archived = [eid for eid in req.email_ids if eid in found]
        for eid in archived:
            found[eid].folder = "archive"
        _ledger.log_action(db, "bulk_archive", None, {"count": len(archived)})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "bulk_archived", "archived": archived, "count": len(archived), "state": state}


@app.post("/create_label")
def create_label(req: CreateLabelRequest):
    with SessionLocal() as db:
        existing = db.query(Label).filter(Label.name == req.name).first()
        if existing:
            return {"status": "error", "message": f"Label '{req.name}' already exists", "state": _get_state_dict(db)}
        lb_id = _ledger.next_id(db, "l")
        db.add(Label(id=lb_id, name=req.name, color=req.color))
        _ledger.log_action(db, "create_label", None, {"name": req.name})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "created", "label_id": lb_id, "name": req.name, "state": state}


@app.post("/empty_trash")
def empty_trash():
    with SessionLocal() as db:
        count = db.query(Email).filter(Email.folder == "trash").count()
        db.query(Email).filter(Email.folder == "trash").delete()
        _ledger.log_action(db, "empty_trash", None, {"deleted_count": count})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "trash_emptied", "deleted_count": count, "state": state}


@app.post("/get_thread")
def get_thread(req: GetThreadRequest):
    with SessionLocal() as db:
        emails = db.query(Email).filter(Email.thread_id == req.thread_id).all()
        emails_sorted = sorted(emails, key=lambda e: e.timestamp)
        state = _get_state_dict(db)
    return {"thread_id": req.thread_id, "emails": [_email_to_dict(e) for e in emails_sorted], "count": len(emails_sorted), "state": state}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
