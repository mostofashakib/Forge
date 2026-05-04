"""Gmail-like Forge RL environment.

Simulates a realistic email client inbox with compose, send, reply, archive,
delete, label, star, and search capabilities. Fully compatible with the Forge
ContainerEpisodeRunner protocol.

Forge endpoints:
  GET  /forge/health  → 200 OK
  GET  /forge/state   → current mailbox state
  POST /forge/reset   → reset to seed state

Action endpoints (all POST, all return updated state):
  /compose    /send      /reply      /archive
  /delete     /mark_read /label      /star       /search
"""
from __future__ import annotations
import copy
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Gmail-like Environment", description="Forge RL environment simulating an email client.")

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED: list[dict] = [
    {
        "id": "e001", "folder": "inbox",
        "from": "alice@company.com", "to": "me@company.com",
        "subject": "Q3 Budget Review",
        "body": "Hi, please review the attached Q3 budget spreadsheet before Friday's meeting.",
        "read": False, "starred": True, "labels": ["work"],
        "timestamp": "2026-05-01T09:15:00Z",
    },
    {
        "id": "e002", "folder": "inbox",
        "from": "bob@company.com", "to": "me@company.com",
        "subject": "Team lunch tomorrow?",
        "body": "Are you free for lunch tomorrow at noon? We're thinking tacos.",
        "read": False, "starred": False, "labels": ["personal"],
        "timestamp": "2026-05-01T10:30:00Z",
    },
    {
        "id": "e003", "folder": "inbox",
        "from": "digest@techdigest.io", "to": "me@company.com",
        "subject": "This week in AI — Top 10 stories",
        "body": "Your weekly AI digest is here. Click to read the full issue.",
        "read": True, "starred": False, "labels": ["newsletter"],
        "timestamp": "2026-05-01T06:00:00Z",
    },
    {
        "id": "e004", "folder": "inbox",
        "from": "carol@company.com", "to": "me@company.com",
        "subject": "Urgent: Production incident",
        "body": "We have a P0 outage. Please join the incident call immediately: meet.company.com/incident",
        "read": False, "starred": True, "labels": ["work", "urgent"],
        "timestamp": "2026-05-01T14:45:00Z",
    },
    {
        "id": "e005", "folder": "inbox",
        "from": "hr@company.com", "to": "me@company.com",
        "subject": "Benefits enrollment reminder",
        "body": "Open enrollment closes May 15th. Log in to update your benefits selection.",
        "read": True, "starred": False, "labels": ["work"],
        "timestamp": "2026-04-30T16:00:00Z",
    },
    {
        "id": "e006", "folder": "inbox",
        "from": "dave@startup.io", "to": "me@company.com",
        "subject": "Partnership opportunity",
        "body": "We'd love to explore a potential partnership. Available for a 30-min call this week?",
        "read": False, "starred": False, "labels": [],
        "timestamp": "2026-04-30T11:20:00Z",
    },
    {
        "id": "e007", "folder": "inbox",
        "from": "noreply@github.com", "to": "me@company.com",
        "subject": "Your repository was starred",
        "body": "user42 starred your repository forge-rl.",
        "read": True, "starred": False, "labels": ["newsletter"],
        "timestamp": "2026-04-29T08:30:00Z",
    },
    {
        "id": "e008", "folder": "inbox",
        "from": "alice@company.com", "to": "me@company.com",
        "subject": "Re: Q3 Budget Review",
        "body": "Thanks for your quick reply. Meeting confirmed for 2pm Friday.",
        "read": False, "starred": False, "labels": ["work"],
        "timestamp": "2026-05-01T15:00:00Z",
    },
    {
        "id": "e009", "folder": "sent",
        "from": "me@company.com", "to": "alice@company.com",
        "subject": "Re: Q3 Budget Review",
        "body": "I've reviewed it. Looks good overall — should discuss a few line items on Friday.",
        "read": True, "starred": False, "labels": [],
        "timestamp": "2026-05-01T09:45:00Z",
    },
    {
        "id": "e010", "folder": "sent",
        "from": "me@company.com", "to": "team@company.com",
        "subject": "Sprint planning notes",
        "body": "Attached are the notes from today's sprint planning. Please review by EOD.",
        "read": True, "starred": False, "labels": ["work"],
        "timestamp": "2026-04-30T17:30:00Z",
    },
    {
        "id": "e011", "folder": "drafts",
        "from": "me@company.com", "to": "boss@company.com",
        "subject": "PTO request — July",
        "body": "Hi, I'd like to request two weeks off in July for a family trip. Is that feasible?",
        "read": True, "starred": False, "labels": [],
        "timestamp": "2026-04-28T12:00:00Z",
    },
]

_LABELS: list[str] = ["work", "personal", "urgent", "newsletter"]

_state: dict = {}


def _reset_state() -> None:
    global _state
    _state = {
        "emails": copy.deepcopy(_SEED),
        "labels": list(_LABELS),
    }


_reset_state()


def _unread_count() -> int:
    return sum(1 for e in _state["emails"] if not e["read"] and e["folder"] == "inbox")


def _by_id(email_id: str) -> dict | None:
    return next((e for e in _state["emails"] if e["id"] == email_id), None)


def _snapshot() -> dict:
    inbox = [e for e in _state["emails"] if e["folder"] == "inbox"]
    return {
        "inbox": inbox,
        "inbox_unread": sum(1 for e in inbox if not e["read"]),
        "sent_count": sum(1 for e in _state["emails"] if e["folder"] == "sent"),
        "draft_count": sum(1 for e in _state["emails"] if e["folder"] == "drafts"),
        "trash_count": sum(1 for e in _state["emails"] if e["folder"] == "trash"),
        "labels": _state["labels"],
        "total_unread": _unread_count(),
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

class ComposeRequest(BaseModel):
    to: str
    subject: str
    body: str = ""


@app.post("/compose", summary="Start composing a new email (creates a draft)")
def compose(req: ComposeRequest):
    draft_id = f"d{uuid.uuid4().hex[:6]}"
    _state["emails"].append({
        "id": draft_id, "folder": "drafts",
        "from": "me@company.com", "to": req.to,
        "subject": req.subject, "body": req.body,
        "read": True, "starred": False, "labels": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "draft_created", "draft_id": draft_id, "state": _snapshot()}


class SendRequest(BaseModel):
    draft_id: str


@app.post("/send", summary="Send a draft email")
def send(req: SendRequest):
    email = _by_id(req.draft_id)
    if not email or email["folder"] != "drafts":
        return {"status": "error", "message": f"Draft '{req.draft_id}' not found", "state": _snapshot()}
    email["folder"] = "sent"
    email["timestamp"] = datetime.now(timezone.utc).isoformat()
    return {"status": "sent", "email_id": req.draft_id, "state": _snapshot()}


class ReplyRequest(BaseModel):
    email_id: str
    body: str


@app.post("/reply", summary="Reply to an email")
def reply(req: ReplyRequest):
    original = _by_id(req.email_id)
    if not original:
        return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _snapshot()}
    reply_id = f"r{uuid.uuid4().hex[:6]}"
    _state["emails"].append({
        "id": reply_id, "folder": "sent",
        "from": "me@company.com", "to": original["from"],
        "subject": f"Re: {original['subject']}", "body": req.body,
        "read": True, "starred": False, "labels": list(original.get("labels", [])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    if not original["read"]:
        original["read"] = True
    return {"status": "replied", "reply_id": reply_id, "state": _snapshot()}


class EmailIdRequest(BaseModel):
    email_id: str


@app.post("/archive", summary="Archive an email (removes from inbox)")
def archive(req: EmailIdRequest):
    email = _by_id(req.email_id)
    if not email:
        return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _snapshot()}
    email["folder"] = "archive"
    return {"status": "archived", "email_id": req.email_id, "state": _snapshot()}


@app.post("/delete", summary="Delete an email (moves to trash)")
def delete(req: EmailIdRequest):
    email = _by_id(req.email_id)
    if not email:
        return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _snapshot()}
    email["folder"] = "trash"
    return {"status": "deleted", "email_id": req.email_id, "state": _snapshot()}


class MarkReadRequest(BaseModel):
    email_id: str
    read: bool = True


@app.post("/mark_read", summary="Mark an email as read or unread")
def mark_read(req: MarkReadRequest):
    email = _by_id(req.email_id)
    if not email:
        return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _snapshot()}
    email["read"] = req.read
    return {"status": "marked", "email_id": req.email_id, "read": req.read, "state": _snapshot()}


class LabelRequest(BaseModel):
    email_id: str
    label: str
    add: bool = True


@app.post("/label", summary="Apply or remove a label from an email")
def label(req: LabelRequest):
    email = _by_id(req.email_id)
    if not email:
        return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _snapshot()}
    if req.add:
        if req.label not in email["labels"]:
            email["labels"].append(req.label)
        if req.label not in _state["labels"]:
            _state["labels"].append(req.label)
    else:
        email["labels"] = [l for l in email["labels"] if l != req.label]
    return {"status": "labeled", "email_id": req.email_id, "label": req.label, "added": req.add, "state": _snapshot()}


class StarRequest(BaseModel):
    email_id: str
    starred: bool = True


@app.post("/star", summary="Star or unstar an email")
def star(req: StarRequest):
    email = _by_id(req.email_id)
    if not email:
        return {"status": "error", "message": f"Email '{req.email_id}' not found", "state": _snapshot()}
    email["starred"] = req.starred
    return {"status": "starred", "email_id": req.email_id, "starred": req.starred, "state": _snapshot()}


class SearchRequest(BaseModel):
    query: str
    folder: str = "all"


@app.post("/search", summary="Search emails by subject, body, or sender")
def search(req: SearchRequest):
    q = req.query.lower()
    emails = _state["emails"]
    if req.folder != "all":
        emails = [e for e in emails if e["folder"] == req.folder]
    results = [
        e for e in emails
        if q in e["subject"].lower() or q in e["body"].lower() or q in e["from"].lower()
    ]
    return {"results": results, "count": len(results), "state": _snapshot()}
