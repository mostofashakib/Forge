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
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import (
    Boolean, Column, Integer, String, Text, create_engine, text
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

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

    id = Column(String, primary_key=True, default=lambda: f"e{uuid.uuid4().hex[:8]}")
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

    id = Column(String, primary_key=True, default=lambda: f"l{uuid.uuid4().hex[:6]}")
    name = Column(String, unique=True, nullable=False)
    color = Column(String, default="#1a73e8")


class SavedState(Base):
    __tablename__ = "saved_states"

    slot = Column(String, primary_key=True)
    data = Column(Text, nullable=False)  # JSON blob


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

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    }


def _dump_full_db(db: Session) -> dict:
    return {
        "emails": [_email_to_dict(e) for e in db.query(Email).all()],
        "contacts": [_contact_to_dict(c) for c in db.query(Contact).all()],
        "labels": [_label_to_dict(lb) for lb in db.query(Label).all()],
    }


def _restore_from_dict(db: Session, data: dict) -> None:
    db.query(Email).delete()
    db.query(Contact).delete()
    db.query(Label).delete()
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
            timestamp=e.get("timestamp", _now()),
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

_SEED_EMAILS = [
    # ---- P0 incident thread ----
    dict(id="e001", thread_id="t001", folder="inbox", from_addr="carol@company.com",
         to_addr="me@company.com", cc="engineering@company.com",
         subject="[P0] Production DB latency spike — 10x normal",
         body=(
             "Hey team,\n\nWe're seeing a 10x latency spike on our primary Postgres cluster. "
             "p99 is at 8 seconds. Dashboard: https://monitoring.company.com/db\n\n"
             "All hands on deck. Join the incident bridge: https://meet.company.com/incident-042\n\n"
             "Carol (On-call SRE)"
         ),
         is_read=False, is_starred=True, labels='["work","urgent"]',
         timestamp="2026-05-04T02:14:00Z", has_attachment=False),

    dict(id="e002", thread_id="t001", folder="inbox", from_addr="bob@company.com",
         to_addr="me@company.com", cc="engineering@company.com",
         subject="Re: [P0] Production DB latency spike — 10x normal",
         body=(
             "I'm investigating. Looks like a missing index after the migration that ran at 01:55 UTC. "
             "Creating the index now — ETA 15 mins. Will update here.\n\nBob"
         ),
         is_read=False, is_starred=False, labels='["work","urgent"]',
         timestamp="2026-05-04T02:31:00Z", has_attachment=False),

    dict(id="e003", thread_id="t001", folder="inbox", from_addr="alice@company.com",
         to_addr="me@company.com", cc="",
         subject="Re: [P0] Production DB latency spike — 10x normal",
         body=(
             "Confirmed. Index created, latency is recovering. p99 back to 420ms. "
             "Post-mortem doc: https://docs.company.com/postmortem-042\n\n"
             "Great catch Bob. Let's schedule a retro for Monday.\n\nAlice"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T02:58:00Z", has_attachment=False),

    # ---- Code review thread ----
    dict(id="e004", thread_id="t002", folder="inbox", from_addr="noreply@github.com",
         to_addr="me@company.com", cc="",
         subject="[forge-rl] PR #312: Add SQLAlchemy persistence layer",
         body=(
             "bob opened pull request #312: Add SQLAlchemy persistence layer\n\n"
             "This PR replaces the in-memory state with a proper SQLAlchemy + SQLite backend. "
             "Includes full snapshot/restore support.\n\n"
             "Review it at: https://github.com/company/forge-rl/pull/312\n\n"
             "— GitHub"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-03T16:20:00Z", has_attachment=False),

    dict(id="e005", thread_id="t002", folder="inbox", from_addr="noreply@github.com",
         to_addr="me@company.com", cc="",
         subject="Re: [forge-rl] PR #312: Add SQLAlchemy persistence layer",
         body=(
             "alice left a review on PR #312:\n\n"
             "> Great work overall! Left a few nits on the session management. "
             "The session should be closed in a finally block to avoid leaks.\n\n"
             "See the review: https://github.com/company/forge-rl/pull/312#review-9981"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-03T17:45:00Z", has_attachment=False),

    # ---- Performance review ----
    dict(id="e006", thread_id="t003", folder="inbox", from_addr="carol@company.com",
         to_addr="me@company.com", cc="",
         subject="Performance Review — Self-Assessment Due May 15",
         body=(
             "Hi,\n\nYour H1 self-assessment is due May 15th in Lattice. "
             "This is your opportunity to highlight your accomplishments, "
             "set goals for H2, and provide upward feedback.\n\n"
             "Link: https://company.latticehq.com/reviews\n\n"
             "Please reach out if you have any questions.\n\nCarol\nHR Business Partner"
         ),
         is_read=True, is_starred=True, labels='["work"]',
         timestamp="2026-05-02T09:00:00Z", has_attachment=False),

    # ---- Meeting request ----
    dict(id="e007", thread_id="t004", folder="inbox", from_addr="alice@company.com",
         to_addr="me@company.com", cc="bob@company.com",
         subject="1:1 sync this week — agenda?",
         body=(
             "Hey,\n\nI want to make sure we cover a few things in our 1:1 Thursday:\n"
             "1. Q2 OKR progress\n"
             "2. The new architecture proposal\n"
             "3. Your thoughts on the on-call rotation\n\n"
             "Does 2pm Thursday still work for you? Let me know if you want to add anything.\n\n"
             "Alice"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-02T14:30:00Z", has_attachment=False),

    # ---- AWS billing alert ----
    dict(id="e008", thread_id="t005", folder="inbox", from_addr="support@aws.com",
         to_addr="me@company.com", cc="",
         subject="AWS Bill Alert: Estimated charges $4,231.00 (threshold: $3,000)",
         body=(
             "Dear AWS Customer,\n\n"
             "Your estimated AWS bill for the current month has exceeded your billing alert threshold.\n\n"
             "Account: 123456789012\n"
             "Current estimated charges: $4,231.00\n"
             "Threshold: $3,000.00\n"
             "Billing period: May 1 – May 4, 2026\n\n"
             "Top services by cost:\n"
             "  EC2: $1,842.00\n"
             "  RDS: $1,107.00\n"
             "  S3:  $482.00\n"
             "  Data Transfer: $800.00\n\n"
             "Review your usage at: https://console.aws.amazon.com/billing\n\n"
             "— AWS Billing"
         ),
         is_read=False, is_starred=True, labels='["work","urgent"]',
         timestamp="2026-05-04T06:00:00Z", has_attachment=False),

    # ---- GitHub notifications ----
    dict(id="e009", thread_id="t006", folder="inbox", from_addr="noreply@github.com",
         to_addr="me@company.com", cc="",
         subject="[forge-rl] New issue: Agent fails to reset state after episode #321",
         body=(
             "eve opened issue #321: Agent fails to reset state after episode\n\n"
             "When the episode ends and reset is called, the DB still contains messages "
             "from the previous episode. Steps to reproduce attached.\n\n"
             "https://github.com/company/forge-rl/issues/321"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-03T11:22:00Z", has_attachment=False),

    dict(id="e010", thread_id="t007", folder="inbox", from_addr="noreply@github.com",
         to_addr="me@company.com", cc="",
         subject="[forge-rl] Your PR #309 was merged",
         body=(
             "Congratulations! alice merged your pull request #309: "
             "Fix CORS headers for embedded environments\n\n"
             "https://github.com/company/forge-rl/pull/309"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-02T18:01:00Z", has_attachment=False),

    # ---- TechCrunch newsletter ----
    dict(id="e011", thread_id="t008", folder="inbox", from_addr="newsletter@techcrunch.com",
         to_addr="me@company.com", cc="",
         subject="TechCrunch Daily: OpenAI unveils o4, Anthropic raises $4B, Meta opens its LLM",
         body=(
             "YOUR DAILY BRIEFING — May 4, 2026\n\n"
             "1. OpenAI launches o4 reasoning model with 'superhuman' math ability\n"
             "2. Anthropic closes $4B Series F at $120B valuation\n"
             "3. Meta open-sources Llama 5 under Apache 2.0\n"
             "4. Mistral releases multilingual embedding model\n"
             "5. EU AI Act enforcement begins — first fines issued\n\n"
             "Read more at https://techcrunch.com/newsletter"
         ),
         is_read=True, is_starred=False, labels='["newsletter"]',
         timestamp="2026-05-04T07:00:00Z", has_attachment=False),

    dict(id="e012", thread_id="t009", folder="inbox", from_addr="newsletter@techcrunch.com",
         to_addr="me@company.com", cc="",
         subject="TechCrunch Weekly: The State of AI Infrastructure",
         body=(
             "WEEKLY DEEP DIVE\n\n"
             "This week we look at the booming AI infrastructure market. "
             "GPU clusters, inference optimization, and the companies betting on "
             "on-prem AI. Plus: is the cloud bubble deflating?\n\n"
             "Read the full issue: https://techcrunch.com/weekly/ai-infra-2026"
         ),
         is_read=False, is_starred=False, labels='["newsletter"]',
         timestamp="2026-05-03T08:00:00Z", has_attachment=False),

    # ---- Bob colleague emails ----
    dict(id="e013", thread_id="t010", folder="inbox", from_addr="bob@company.com",
         to_addr="me@company.com", cc="",
         subject="Lunch Tuesday?",
         body=(
             "Hey! Free for lunch Tuesday? The new ramen place opened on 5th Ave, "
             "want to check it out. Around 12:30?\n\nBob"
         ),
         is_read=True, is_starred=False, labels='["personal"]',
         timestamp="2026-05-01T11:15:00Z", has_attachment=False),

    dict(id="e014", thread_id="t011", folder="inbox", from_addr="bob@company.com",
         to_addr="me@company.com", cc="",
         subject="Re: Sprint planning — story points question",
         body=(
             "Agreed. I think we should cap the sprint at 40 points given the on-call load. "
             "The database migration ticket alone will take 2 days minimum. "
             "Want to sync before the planning session?\n\nBob"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-03T15:30:00Z", has_attachment=False),

    # ---- Alice manager emails ----
    dict(id="e015", thread_id="t012", folder="inbox", from_addr="alice@company.com",
         to_addr="me@company.com", cc="",
         subject="Promo packet — deadline extended to May 20",
         body=(
             "Hi,\n\nGood news — the promotion committee has extended the promo packet "
             "submission deadline to May 20th. Please make sure to include:\n"
             "- Impact summary (1 page max)\n"
             "- Peer feedback references\n"
             "- Engineering metrics (PRs, incidents handled, projects led)\n\n"
             "I'm happy to review a draft. Let me know.\n\nAlice"
         ),
         is_read=False, is_starred=True, labels='["work"]',
         timestamp="2026-05-02T16:00:00Z", has_attachment=False),

    dict(id="e016", thread_id="t013", folder="inbox", from_addr="alice@company.com",
         to_addr="me@company.com", cc="",
         subject="Team offsite — venue vote",
         body=(
             "Hi team,\n\nI've narrowed down the Q3 offsite to three venues:\n"
             "A) Lake Tahoe (nature-focused, hiking + workshops)\n"
             "B) Napa Valley (wine + team dinners + strategy sessions)\n"
             "C) Austin TX (tech scene + BBQ + SXSW overlap)\n\n"
             "Please vote by Friday: https://forms.company.com/offsite-vote\n\nAlice"
         ),
         is_read=True, is_starred=False, labels='["work","personal"]',
         timestamp="2026-05-01T13:00:00Z", has_attachment=False),

    # ---- Archive / older emails ----
    dict(id="e017", thread_id="t014", folder="archive", from_addr="carol@company.com",
         to_addr="me@company.com", cc="",
         subject="Welcome to the team!",
         body=(
             "Hi! Welcome aboard. We're so excited to have you join the engineering team. "
             "Your first day is Monday May 1st. Please bring your laptop and a valid ID. "
             "Your buddy is Bob — he'll meet you at reception at 9am.\n\nCarol\nHR"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-04-15T10:00:00Z", has_attachment=True),

    dict(id="e018", thread_id="t015", folder="archive", from_addr="support@aws.com",
         to_addr="me@company.com", cc="",
         subject="Your AWS support case #902833 has been resolved",
         body=(
             "Dear customer,\n\n"
             "Your support case #902833 (EC2 instance not reachable in ap-southeast-1) "
             "has been resolved. The root cause was an AZ-level networking issue that "
             "was mitigated at 14:32 UTC.\n\nThank you for your patience.\n\n— AWS Support"
         ),
         is_read=True, is_starred=False, labels='[]',
         timestamp="2026-04-20T15:00:00Z", has_attachment=False),

    # ---- Sent emails ----
    dict(id="e019", thread_id="t001", folder="sent", from_addr="me@company.com",
         to_addr="carol@company.com", cc="engineering@company.com",
         subject="Re: [P0] Production DB latency spike — 10x normal",
         body=(
             "On it. Joining the bridge now. I'll look at slow query logs.\n\nThanks for the quick alert."
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T02:22:00Z", has_attachment=False),

    dict(id="e020", thread_id="t011", folder="sent", from_addr="me@company.com",
         to_addr="bob@company.com", cc="",
         subject="Sprint planning — story points question",
         body=(
             "Hey Bob, quick question for tomorrow's planning — "
             "are we estimating the DB migration as one ticket or breaking it down? "
             "Asking because it changes how I'd size the surrounding tasks.\n\nThanks"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-03T14:55:00Z", has_attachment=False),

    dict(id="e021", thread_id="t004", folder="sent", from_addr="me@company.com",
         to_addr="alice@company.com", cc="",
         subject="Re: 1:1 sync this week — agenda?",
         body=(
             "Hi Alice,\n\nThursday 2pm works great. I'd also like to discuss:\n"
             "- The proposal to move to async workers\n"
             "- Feedback on the on-call docs I drafted\n\nSee you then!"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-02T15:10:00Z", has_attachment=False),

    # ---- Drafts ----
    dict(id="e022", thread_id=None, folder="drafts", from_addr="me@company.com",
         to_addr="carol@company.com", cc="",
         subject="PTO Request — June 16-27",
         body=(
             "Hi Carol,\n\nI'd like to request PTO from June 16th through June 27th (2 weeks). "
             "I've confirmed coverage with Bob for the first week and Alice for the second.\n\n"
             "Please let me know if this works.\n\nThanks,"
         ),
         is_read=True, is_starred=False, labels='[]',
         timestamp="2026-05-03T20:00:00Z", has_attachment=False),

    dict(id="e023", thread_id=None, folder="drafts", from_addr="me@company.com",
         to_addr="boss@company.com", cc="",
         subject="Architecture proposal — async task queue",
         body=(
             "Hi,\n\nI've been working on a proposal to replace our synchronous "
             "task processing with an async queue (Redis + Celery). "
             "Key benefits: 10x throughput, no more timeout errors, better observability.\n\n"
             "Draft doc: [link]\n\nWould love 30 mins to walk through it."
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-02T22:00:00Z", has_attachment=False),

    # ---- Trash ----
    dict(id="e024", thread_id=None, folder="trash", from_addr="promo@deals.example.com",
         to_addr="me@company.com", cc="",
         subject="🔥 Limited time: 70% off all courses this weekend only!",
         body=(
             "FLASH SALE! 70% off everything. Use code FLASH70 at checkout. "
             "Expires Sunday midnight. Thousands of courses, one low price."
         ),
         is_read=True, is_starred=False, labels='[]',
         timestamp="2026-05-01T07:00:00Z", has_attachment=False),
]

_SEED_CONTACTS = [
    dict(email="alice@company.com", name="Alice Chen", avatar_initial="A"),
    dict(email="bob@company.com", name="Bob Martinez", avatar_initial="B"),
    dict(email="carol@company.com", name="Carol Smith", avatar_initial="C"),
    dict(email="noreply@github.com", name="GitHub", avatar_initial="G"),
    dict(email="newsletter@techcrunch.com", name="TechCrunch", avatar_initial="T"),
    dict(email="support@aws.com", name="AWS Support", avatar_initial="A"),
    dict(email="me@company.com", name="Me", avatar_initial="M"),
]

_SEED_LABELS = [
    dict(id="lb01", name="work", color="#1a73e8"),
    dict(id="lb02", name="urgent", color="#d93025"),
    dict(id="lb03", name="personal", color="#188038"),
    dict(id="lb04", name="newsletter", color="#e37400"),
]


def _seed_if_empty() -> None:
    with SessionLocal() as db:
        if db.query(Email).count() > 0:
            return
        for e in _SEED_EMAILS:
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
        for c in _SEED_CONTACTS:
            db.add(Contact(**c))
        for lb in _SEED_LABELS:
            db.add(Label(**lb))
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
        db.query(Email).delete()
        db.query(Contact).delete()
        db.query(Label).delete()
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


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@app.post("/compose")
def compose(req: ComposeRequest):
    draft_id = f"d{uuid.uuid4().hex[:8]}"
    ts = _now()
    with SessionLocal() as db:
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
        db.commit()
        state = _get_state_dict(db)
    return {"status": "draft_created", "draft_id": draft_id, "state": state}


@app.post("/send")
def send(req: SendRequest):
    with SessionLocal() as db:
        email = db.query(Email).filter(Email.id == req.draft_id).first()
        if not email or email.folder != "drafts":
            return {"status": "error", "message": f"Draft '{req.draft_id}' not found", "state": _get_state_dict(db)}
        email.folder = "sent"
        email.timestamp = _now()
        db.commit()
        state = _get_state_dict(db)
    return {"status": "sent", "email_id": req.draft_id, "state": state}


@app.post("/reply")
def reply(req: ReplyRequest):
    with SessionLocal() as db:
        original = db.query(Email).filter(Email.id == req.email_id).first()
        if not original:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        reply_id = f"r{uuid.uuid4().hex[:8]}"
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
            timestamp=_now(),
            has_attachment=False,
        ))
        original.is_read = True
        db.commit()
        state = _get_state_dict(db)
    return {"status": "replied", "reply_id": reply_id, "state": state}


@app.post("/forward")
def forward(req: ForwardRequest):
    with SessionLocal() as db:
        original = db.query(Email).filter(Email.id == req.email_id).first()
        if not original:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        fwd_id = f"f{uuid.uuid4().hex[:8]}"
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
            timestamp=_now(),
            has_attachment=original.has_attachment,
        ))
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
        db.commit()
        state = _get_state_dict(db)
    return {"status": "archived", "email_id": req.email_id, "state": state}


@app.post("/delete")
def delete(req: EmailIdRequest):
    with SessionLocal() as db:
        email = db.query(Email).filter(Email.id == req.email_id).first()
        if not email:
            return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _get_state_dict(db)}
        email.folder = "trash"
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
                db.add(Label(id=f"l{uuid.uuid4().hex[:6]}", name=req.label, color="#1a73e8"))
        else:
            current = [l for l in current if l != req.label]
        email.labels = json.dumps(current)
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
        db.commit()
        state = _get_state_dict(db)
    return {"status": "moved", "email_id": req.email_id, "folder": req.folder, "state": state}


@app.post("/bulk_archive")
def bulk_archive(req: BulkArchiveRequest):
    archived = []
    with SessionLocal() as db:
        for eid in req.email_ids:
            email = db.query(Email).filter(Email.id == eid).first()
            if email:
                email.folder = "archive"
                archived.append(eid)
        db.commit()
        state = _get_state_dict(db)
    return {"status": "bulk_archived", "archived": archived, "count": len(archived), "state": state}


@app.post("/create_label")
def create_label(req: CreateLabelRequest):
    with SessionLocal() as db:
        existing = db.query(Label).filter(Label.name == req.name).first()
        if existing:
            return {"status": "error", "message": f"Label '{req.name}' already exists", "state": _get_state_dict(db)}
        lb_id = f"l{uuid.uuid4().hex[:6]}"
        db.add(Label(id=lb_id, name=req.name, color=req.color))
        db.commit()
        state = _get_state_dict(db)
    return {"status": "created", "label_id": lb_id, "name": req.name, "state": state}


@app.post("/empty_trash")
def empty_trash():
    with SessionLocal() as db:
        count = db.query(Email).filter(Email.folder == "trash").count()
        db.query(Email).filter(Email.folder == "trash").delete()
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
