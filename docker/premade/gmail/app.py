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


def _log_action(db: Session, action_type: str, target_id: str = None, payload: dict = None) -> None:
    db.add(ActionLog(
        id=f"a{uuid.uuid4().hex[:8]}",
        action_type=action_type,
        target_id=target_id,
        payload=json.dumps(payload or {}),
        timestamp=_now(),
    ))


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
        "action_log": [
            {"id": a.id, "action_type": a.action_type, "target_id": a.target_id,
             "payload": a.payload, "timestamp": a.timestamp}
            for a in db.query(ActionLog).order_by(ActionLog.timestamp).all()
        ],
    }


def _restore_from_dict(db: Session, data: dict) -> None:
    db.query(Email).delete()
    db.query(Contact).delete()
    db.query(Label).delete()
    db.query(ActionLog).delete()
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
    for a in data.get("action_log", []):
        db.add(ActionLog(id=a["id"], action_type=a["action_type"], target_id=a.get("target_id"),
                         payload=a.get("payload", "{}"), timestamp=a["timestamp"]))
    db.commit()


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_EMAILS = [
    # ── THREAD t001: P0 incident (5 inbox + 1 sent) ──────────────────────────
    dict(id="e001", thread_id="t001", folder="inbox",
         from_addr="pagerduty@pagerduty.com", to_addr="me@company.com",
         cc="engineering@company.com",
         subject="[ALERT] P0 — Production DB latency spike (p99 > 8s)",
         body=(
             "PAGERDUTY ALERT — Severity: P0\n\n"
             "Alert: api-gateway — Postgres p99 > 5000ms\n"
             "Current: 8,340 ms  |  Threshold: 500 ms\n"
             "Started: 2026-05-04 02:11 UTC\n"
             "Affected: api-gateway, user-service, billing-service\n\n"
             "Dashboard: https://monitoring.company.com/db/latency\n"
             "Runbook:   https://docs.company.com/runbooks/db-latency\n"
             "On-call:   carol@company.com\n\n"
             "Acknowledge: https://pagerduty.com/incidents/P042\n\n"
             "— PagerDuty"
         ),
         is_read=True, is_starred=True, labels='["work","urgent"]',
         timestamp="2026-05-04T02:11:00Z", has_attachment=False),

    dict(id="e002", thread_id="t001", folder="inbox",
         from_addr="carol@company.com", to_addr="me@company.com",
         cc="engineering@company.com",
         subject="Re: [ALERT] P0 — Production DB latency spike",
         body=(
             "All — p99 is at 8.3 s on the primary cluster. Hitting everything that touches the users table.\n\n"
             "Bridge: https://meet.company.com/incident-042\n"
             "Doc:    https://docs.company.com/incidents/042\n\n"
             "What I know:\n"
             "• Started ~02:09 UTC right after the scheduled migration\n"
             "• CPU on db-primary-1 at 98%\n"
             "• Tons of sequential scans on users table\n\n"
             "Need someone on slow query logs NOW.\n\nCarol"
         ),
         is_read=True, is_starred=True, labels='["work","urgent"]',
         timestamp="2026-05-04T02:14:00Z", has_attachment=False),

    dict(id="e003", thread_id="t001", folder="inbox",
         from_addr="bob@company.com", to_addr="me@company.com",
         cc="engineering@company.com",
         subject="Re: [ALERT] P0 — Production DB latency spike",
         body=(
             "Found it. The 01:55 migration dropped and recreated users.email index "
             "but forgot the composite (user_id, created_at) index billing depends on.\n\n"
             "pg_stat_user_indexes confirms it's missing. Creating concurrently now — "
             "won't lock the table. ETA 12-15 min.\n\n"
             "Running:\n"
             "  CREATE INDEX CONCURRENTLY idx_users_id_created\n"
             "  ON users(user_id, created_at);\n\n"
             "Will update here when done.\n\nBob"
         ),
         is_read=True, is_starred=False, labels='["work","urgent"]',
         timestamp="2026-05-04T02:31:00Z", has_attachment=False),

    dict(id="e004", thread_id="t001", folder="inbox",
         from_addr="alice@company.com", to_addr="me@company.com",
         cc="engineering@company.com",
         subject="Re: [ALERT] P0 — RESOLVED — latency normal",
         body=(
             "Index creation complete. Confirming recovery:\n"
             "  p99:  8340 ms → 420 ms  ✓\n"
             "  CPU (db-primary-1): 98% → 12%  ✓\n"
             "  Sequential scans on users: 0  ✓\n"
             "  All services green  ✓\n\n"
             "Incident closed 02:58 UTC. Total impact: ~49 minutes.\n\n"
             "Next steps:\n"
             "1. Bob — add missing index to rollback script\n"
             "2. Everyone — post-mortem Monday 10 AM: https://docs.company.com/postmortem-042\n"
             "3. We need a pre-migration index audit checklist\n\n"
             "Great work everyone.\n\nAlice"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T02:58:00Z", has_attachment=False),

    dict(id="e005", thread_id="t001", folder="inbox",
         from_addr="carol@company.com", to_addr="me@company.com",
         cc="",
         subject="Re: [ALERT] P0 — alerting threshold follow-up",
         body=(
             "PagerDuty incident closed. Timeline written up in the doc.\n\n"
             "One flag: our alert fires at 500 ms but actual business impact for billing "
             "starts around 200 ms. Can someone look at tightening that? "
             "I'll add it as a task in the post-mortem.\n\n"
             "Also adding 'migration index audit' to our deploy checklist this week.\n\nCarol"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T03:15:00Z", has_attachment=False),

    # ── THREAD t002: PR review (4 messages) ───────────────────────────────────
    dict(id="e006", thread_id="t002", folder="inbox",
         from_addr="noreply@github.com", to_addr="me@company.com",
         cc="",
         subject="[forge-rl] PR #312: Add SQLAlchemy persistence layer",
         body=(
             "eve opened pull request #312\n\n"
             "Add SQLAlchemy persistence layer\n\n"
             "Replaces the in-memory state dict with SQLAlchemy + SQLite for proper "
             "persistence across restarts and full snapshot/restore support.\n\n"
             "Changes:\n"
             "  • Models: Environment, Episode, StepRecord, SavedState\n"
             "  • SessionLocal factory with connection pooling\n"
             "  • /forge/snapshot and /forge/restore endpoints\n"
             "  • Migration script for existing deployments\n"
             "  • 47 new unit tests — all passing\n\n"
             "Closes #298, #301\n"
             "https://github.com/company/forge-rl/pull/312\n\n— GitHub"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-03T16:20:00Z", has_attachment=False),

    dict(id="e007", thread_id="t002", folder="inbox",
         from_addr="noreply@github.com", to_addr="me@company.com",
         cc="",
         subject="Re: [forge-rl] PR #312 — alice requested changes",
         body=(
             "alice left a review — Changes requested\n\n"
             "> Two things before merging:\n"
             "> 1. Session must be closed in a `finally` block — leaks under error conditions.\n"
             "> 2. Snapshot serializer doesn't handle circular refs. If Episode has a\n"
             ">    back-reference to Environment this will blow up.\n"
             ">\n"
             "> Otherwise LGTM. Approving once these are fixed.\n\n"
             "https://github.com/company/forge-rl/pull/312#review-9981\n\n— GitHub"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-03T17:45:00Z", has_attachment=False),

    dict(id="e008", thread_id="t002", folder="inbox",
         from_addr="noreply@github.com", to_addr="me@company.com",
         cc="",
         subject="Re: [forge-rl] PR #312 — bob requested changes",
         body=(
             "bob left a review — Changes requested\n\n"
             "> Migration script assumes SQLite but docs say we support Postgres.\n"
             "> `check_same_thread=False` is SQLite-only — will error on Postgres.\n"
             ">\n"
             "> Minor: SEED_DATA is 847 lines inline. Consider a JSON file loaded at startup.\n\n"
             "https://github.com/company/forge-rl/pull/312#review-9994\n\n— GitHub"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T09:30:00Z", has_attachment=False),

    dict(id="e009", thread_id="t002", folder="inbox",
         from_addr="noreply@github.com", to_addr="me@company.com",
         cc="",
         subject="Re: [forge-rl] PR #312 — approved and merged",
         body=(
             "alice approved PR #312 — Add SQLAlchemy persistence layer\n\n"
             "> Session management looks good. Finally block is correct.\n"
             "> Circular ref issue handled. LGTM — merging after CI.\n\n"
             "✅ All 3 checks passed\n"
             "✅ 2 approvals (alice, carol)\n"
             "🎉 Merged into main\n\n"
             "https://github.com/company/forge-rl/pull/312\n\n— GitHub"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T14:10:00Z", has_attachment=False),

    # ── THREAD t003: Performance review ───────────────────────────────────────
    dict(id="e010", thread_id="t003", folder="inbox",
         from_addr="carol@company.com", to_addr="me@company.com",
         cc="",
         subject="H1 Self-Assessment — Due May 15 (Action Required)",
         body=(
             "Hi,\n\n"
             "Your H1 self-assessment is due in Lattice by May 15th — required before "
             "your review meeting with Alice.\n\n"
             "What to include:\n"
             "  • Top 3 accomplishments with measurable impact\n"
             "  • Progress against H1 goals\n"
             "  • One area of significant growth\n"
             "  • H2 goals\n"
             "  • (Optional) Upward feedback for your manager\n\n"
             "Lattice: https://company.latticehq.com/reviews/h1-2026\n"
             "Time estimate: 30-45 min. Bullet points are fine.\n\n"
             "Carol\nHR Business Partner"
         ),
         is_read=True, is_starred=True, labels='["work"]',
         timestamp="2026-05-02T09:00:00Z", has_attachment=False),

    dict(id="e011", thread_id="t003", folder="inbox",
         from_addr="carol@company.com", to_addr="me@company.com",
         cc="",
         subject="Re: H1 Self-Assessment — 5 days left",
         body=(
             "Quick reminder — H1 self-assessments are due in 5 days (May 15).\n\n"
             "About 40% of the team has submitted. You haven't yet.\n\n"
             "If you want a prep call, Alice has 15-min slots available this week.\n\nCarol"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-10T09:00:00Z", has_attachment=False),

    # ── THREAD t004: 1:1 with manager ────────────────────────────────────────
    dict(id="e012", thread_id="t004", folder="inbox",
         from_addr="alice@company.com", to_addr="me@company.com",
         cc="",
         subject="1:1 Thursday — agenda?",
         body=(
             "Hey,\n\n"
             "Looking forward to our 1:1 Thursday 2pm. My agenda:\n"
             "  1. Q2 OKR review — we're at 74%, want to talk through blockers\n"
             "  2. Async worker architecture proposal\n"
             "  3. On-call rotation sustainability\n"
             "  4. H2 career development goals\n\n"
             "Anything you'd like to add? Let me know if 2pm doesn't work.\n\nAlice"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-06T14:30:00Z", has_attachment=False),

    # ── THREAD t005: AWS billing ───────────────────────────────────────────────
    dict(id="e013", thread_id="t005", folder="inbox",
         from_addr="billing-alerts@aws.amazon.com", to_addr="me@company.com",
         cc="",
         subject="AWS Budget Alert: 'Production' exceeded $3,000 — currently $4,312",
         body=(
             "Your AWS Budget 'Production' has exceeded its monthly threshold.\n\n"
             "Budget:  Production\n"
             "Account: 123456789012\n"
             "Threshold: $3,000.00\n"
             "Current:   $4,312.47  (May 1–4, only 4 days in)\n"
             "Projected: ~$33,000 this month (approved: $18,000)\n\n"
             "Breakdown:\n"
             "  EC2:            $1,892\n"
             "  RDS:            $1,140\n"
             "  Data Transfer:  $822\n"
             "  S3:             $458\n\n"
             "Review: https://console.aws.amazon.com/billing\n\n"
             "— AWS Billing & Cost Management"
         ),
         is_read=False, is_starred=True, labels='["work","urgent"]',
         timestamp="2026-05-04T06:00:00Z", has_attachment=False),

    # ── THREAD t006: Jira ticket assigned ─────────────────────────────────────
    dict(id="e014", thread_id="t006", folder="inbox",
         from_addr="jira@company.atlassian.net", to_addr="me@company.com",
         cc="",
         subject="[FORGE-421] Episode timeout handling — assigned to you",
         body=(
             "Ticket assigned to you by alice\n\n"
             "FORGE-421 — Implement episode timeout handling\n"
             "Type: Feature  |  Priority: High  |  Sprint: 24  |  Points: 5\n\n"
             "Description:\n"
             "Episodes can run indefinitely if the agent loops. Need a configurable "
             "max_steps that terminates the episode gracefully with a partial reward.\n\n"
             "Acceptance criteria:\n"
             "  • max_steps configurable per-episode via /run\n"
             "  • status=timeout when steps >= max_steps\n"
             "  • Partial reward calculated and returned\n"
             "  • Timeout events appear in observability feed\n\n"
             "https://company.atlassian.net/browse/FORGE-421"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-05T09:00:00Z", has_attachment=False),

    # ── THREAD t007: Zoom recording ────────────────────────────────────────────
    dict(id="e015", thread_id="t007", folder="inbox",
         from_addr="no-reply@zoom.us", to_addr="me@company.com",
         cc="",
         subject="Recording available: Engineering All-Hands — May 3, 2026",
         body=(
             "Your Zoom cloud recording is ready.\n\n"
             "Meeting: Engineering All-Hands\n"
             "Date:    May 3, 2026  |  Duration: 1h 12m\n"
             "Passcode: Eng2026!\n\n"
             "Watch: https://zoom.us/rec/share/abc123  (available 30 days)\n\n"
             "Topics (from auto-transcript):\n"
             "  • Q1 retrospective and lessons learned\n"
             "  • New incident response process\n"
             "  • Architecture roadmap H2 2026\n"
             "  • Open Q&A\n\n"
             "— Zoom"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-03T20:00:00Z", has_attachment=False),

    # ── THREAD t008: LinkedIn ──────────────────────────────────────────────────
    dict(id="e016", thread_id="t008", folder="inbox",
         from_addr="messages-noreply@linkedin.com", to_addr="me@company.com",
         cc="",
         subject="3 connection requests + 1 new message on LinkedIn",
         body=(
             "Activity on LinkedIn:\n\n"
             "CONNECTION REQUESTS (3)\n"
             "  • Sarah Chen — VP Engineering at Acme Corp\n"
             "  • Marcus Webb — Engineering Manager at Stripe\n"
             "  • Priya Nair — Senior SWE at OpenAI\n\n"
             "NEW MESSAGE — Marcus Webb:\n"
             "  \"Hey! We met at the distributed systems meetup last month. "
             "I'm building out our infra team and thought of you — want to grab "
             "coffee and catch up? Let me know!\"\n\n"
             "https://linkedin.com/notifications\n\n"
             "— LinkedIn"
         ),
         is_read=True, is_starred=False, labels='["personal"]',
         timestamp="2026-05-04T07:30:00Z", has_attachment=False),

    # ── THREAD t009: Notion digest ─────────────────────────────────────────────
    dict(id="e017", thread_id="t009", folder="inbox",
         from_addr="notify@notion.so", to_addr="me@company.com",
         cc="",
         subject="Notion: 7 updates in spaces you follow",
         body=(
             "What's new this week:\n\n"
             "ENGINEERING WIKI (4 updates)\n"
             "  alice  — 'System Architecture Overview' — added async worker diagram\n"
             "  bob    — 'DB Migration Checklist v2' — post-incident addition\n"
             "  carol  — 'Incident Response Runbook' — tighter alert thresholds\n"
             "  eve    — 'SQLAlchemy Best Practices' — session management guide\n\n"
             "PRODUCT ROADMAP (2 updates)\n"
             "  carol  — 'Q3 Roadmap' — added AI search spec\n"
             "  dave   — 'Dashboard Redesign' — Figma mockup links added\n\n"
             "COMPANY (1 update)\n"
             "  alice  — '4-Day Work Week FAQ' — answers for the June pilot\n\n"
             "https://notion.so/company/home\n\n— Notion"
         ),
         is_read=True, is_starred=False, labels='["newsletter"]',
         timestamp="2026-05-04T08:00:00Z", has_attachment=False),

    # ── THREAD t010: Lunch ────────────────────────────────────────────────────
    dict(id="e018", thread_id="t010", folder="inbox",
         from_addr="bob@company.com", to_addr="me@company.com",
         cc="",
         subject="Lunch Tuesday? New ramen place on 5th",
         body=(
             "Hey,\n\n"
             "New ramen place on 5th Ave finally opened — walked by and the line was "
             "around the block. Must be good.\n\n"
             "Tuesday 12:30? We could sync on the sprint and save ourselves a meeting.\n\nBob"
         ),
         is_read=True, is_starred=False, labels='["personal"]',
         timestamp="2026-05-01T11:15:00Z", has_attachment=False),

    # ── THREAD t011: Sprint tickets ───────────────────────────────────────────
    dict(id="e019", thread_id="t011", folder="inbox",
         from_addr="jira@company.atlassian.net", to_addr="me@company.com",
         cc="",
         subject="Sprint 24 started — 8 tickets assigned to you",
         body=(
             "Sprint 24 active: May 5–18, 2026\n\n"
             "YOUR TICKETS (27 points total):\n"
             "  FORGE-421  Episode timeout handling           5 pts  High\n"
             "  FORGE-415  Fix race condition in runner       3 pts  Medium\n"
             "  FORGE-408  Better action error messages       2 pts  Low\n"
             "  FORGE-402  Bulk snapshot endpoint             3 pts  Medium\n"
             "  FORGE-398  Forge protocol v2 docs             2 pts  Low\n"
             "  FORGE-395  Upgrade FastAPI to 0.115           1 pt   Low\n"
             "  FORGE-388  Rate limiting on /run              3 pts  Medium\n"
             "  FORGE-371  Agent observation format v2        8 pts  High\n\n"
             "Sprint velocity target: 30 pts\n"
             "https://company.atlassian.net/jira/boards/24"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-05T09:30:00Z", has_attachment=False),

    dict(id="e020", thread_id="t011", folder="inbox",
         from_addr="bob@company.com", to_addr="me@company.com",
         cc="",
         subject="Re: Sprint 24 — your load looks heavy",
         body=(
             "Hey — 27 pts plus on-call is a lot. FORGE-371 will probably be more "
             "than 8 pts once we get into the v2 schema design.\n\n"
             "Want to pair on it? I have thoughts on cutting scope while keeping "
             "backwards compat. DM me a time.\n\nBob"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-05T15:30:00Z", has_attachment=False),

    # ── THREAD t012: Promo packet ────────────────────────────────────────────
    dict(id="e021", thread_id="t012", folder="inbox",
         from_addr="alice@company.com", to_addr="me@company.com",
         cc="",
         subject="Promotion packet — deadline May 20, tips inside",
         body=(
             "Hi,\n\n"
             "Deadline extended to May 20. Here's what makes a strong packet:\n\n"
             "WHAT THE COMMITTEE LOOKS FOR:\n"
             "  • Concrete impact with numbers ('reduced p99 by 40%')\n"
             "  • Scope of ownership — did you drive a project end-to-end?\n"
             "  • Cross-functional influence — decisions beyond your team?\n"
             "  • Technical depth — already operating at the next level?\n\n"
             "TIPS:\n"
             "  • Don't be modest — this is not the time\n"
             "  • Link to PRs, docs, and incident reports as evidence\n"
             "  • The May 4 incident recovery is a strong example\n\n"
             "Happy to do a review session before you submit. Want 30 min this week?\n\nAlice"
         ),
         is_read=False, is_starred=True, labels='["work"]',
         timestamp="2026-05-06T16:00:00Z", has_attachment=False),

    # ── THREAD t013: Team offsite (2 messages) ───────────────────────────────
    dict(id="e022", thread_id="t013", folder="inbox",
         from_addr="alice@company.com", to_addr="me@company.com",
         cc="engineering@company.com",
         subject="Q3 offsite venue vote — closes Friday",
         body=(
             "Hi team,\n\n"
             "Three options for the Q3 offsite:\n\n"
             "A) Lake Tahoe    Jun 23-25  ~$650/person  Kayaking, hiking, lodge workshops\n"
             "B) Napa Valley   Jul 7-9    ~$850/person  Wineries, team dinners, strategy\n"
             "C) Austin TX     Jul 14-16  ~$750/person  Tech scene, BBQ, conference overlap\n\n"
             "Vote by Friday: https://forms.company.com/offsite-q3-2026\n"
             "Hotel blocks held until May 12.\n\nAlice"
         ),
         is_read=True, is_starred=False, labels='["work","personal"]',
         timestamp="2026-05-01T13:00:00Z", has_attachment=False),

    dict(id="e023", thread_id="t013", folder="inbox",
         from_addr="carol@company.com", to_addr="me@company.com",
         cc="engineering@company.com",
         subject="Re: Q3 offsite — Lake Tahoe wins!",
         body=(
             "Results:\n"
             "  A) Lake Tahoe  7 votes  ⭐ WINNER\n"
             "  B) Napa        4 votes\n"
             "  C) Austin      2 votes\n\n"
             "June 23-25 it is! I'll send hotel booking link by end of week. "
             "Please book by May 20 to get the group rate.\n\nCarol"
         ),
         is_read=False, is_starred=False, labels='["work","personal"]',
         timestamp="2026-05-05T11:00:00Z", has_attachment=False),

    # ── THREAD t014: Figma comment ────────────────────────────────────────────
    dict(id="e024", thread_id="t014", folder="inbox",
         from_addr="noreply@figma.com", to_addr="me@company.com",
         cc="",
         subject="dave mentioned you in a Figma comment",
         body=(
             "dave commented on 'Dashboard Redesign v2 — Engineering Handoff'\n"
             "Page: Observability Panel\n\n"
             "\"@me — can you review the data model for the event timeline? "
             "I designed assuming events have a max duration of 24h but the current "
             "API doesn't seem to enforce that. Will this cause rendering issues?\"\n\n"
             "https://figma.com/file/acme/dashboard-v2?comment=847\n\n— Figma"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T13:30:00Z", has_attachment=False),

    # ── THREAD t015: GitHub issue ──────────────────────────────────────────────
    dict(id="e025", thread_id="t015", folder="inbox",
         from_addr="noreply@github.com", to_addr="me@company.com",
         cc="",
         subject="[forge-rl] Issue #321: action_log not cleared on /forge/reset",
         body=(
             "eve opened issue #321\n\n"
             "action_log entries persist after /forge/reset\n\n"
             "Steps to reproduce:\n"
             "  1. Run an episode\n"
             "  2. POST /forge/reset\n"
             "  3. GET /forge/state\n"
             "  4. action_log is not empty ← bug\n\n"
             "Expected: action_log cleared on reset\n"
             "Actual: previous episode's log persists\n\n"
             "Blocking accurate reward calculation in the eval harness.\n\n"
             "https://github.com/company/forge-rl/issues/321"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T11:22:00Z", has_attachment=False),

    # ── THREAD t016: TLDR newsletters ─────────────────────────────────────────
    dict(id="e026", thread_id="t016", folder="inbox",
         from_addr="digest@tldr.tech", to_addr="me@company.com",
         cc="",
         subject="TLDR — Claude 4, PostgreSQL 18, OAuth vulnerability",
         body=(
             "TLDR | May 4, 2026\n\n"
             "BIG TECH\n"
             "  • Anthropic launches Claude 4 with extended thinking — passes bar exam at 98th pct\n"
             "  • OpenAI releases o4-mini at 80% cost reduction for API customers\n"
             "  • Google DeepMind announces Gemini Ultra 2 with native code execution\n\n"
             "PROGRAMMING\n"
             "  • Python 3.14 final: t-string literals, free-threaded mode stable\n"
             "  • Node 24 LTS: native TypeScript support, faster startup\n"
             "  • PostgreSQL 18 beta: columnar storage, Parquet export, 3x analytics\n\n"
             "SECURITY\n"
             "  • Critical OAuth PKCE bypass affects 40+ providers — patch immediately\n"
             "  • nginx 1.27.3 patches 0-day — update now\n\n"
             "https://tldr.tech"
         ),
         is_read=True, is_starred=False, labels='["newsletter"]',
         timestamp="2026-05-04T07:15:00Z", has_attachment=False),

    dict(id="e027", thread_id="t016", folder="inbox",
         from_addr="digest@tldr.tech", to_addr="me@company.com",
         cc="",
         subject="TLDR — Deno 2.3, React 20 RC, fusion milestone",
         body=(
             "TLDR | May 3, 2026\n\n"
             "PROGRAMMING\n"
             "  • Deno 2.3 reaches Node.js API parity — 94% of npm packages compatible\n"
             "  • React 20 RC: server components stable, concurrent transitions by default\n"
             "  • Rust 1.88: async closures stabilized\n\n"
             "SCIENCE\n"
             "  • Sustained net-energy fusion for 4th consecutive quarter\n"
             "  • AlphaFold 4 predicts multi-protein complexes at near-experimental accuracy\n\n"
             "https://tldr.tech"
         ),
         is_read=False, is_starred=False, labels='["newsletter"]',
         timestamp="2026-05-03T07:15:00Z", has_attachment=False),

    # ── THREAD t017: Calendar invite ───────────────────────────────────────────
    dict(id="e028", thread_id="t017", folder="inbox",
         from_addr="calendar-notification@google.com", to_addr="me@company.com",
         cc="",
         subject="Invitation: Post-mortem retro @ Mon May 6, 10 AM",
         body=(
             "You have been invited:\n\n"
             "Post-mortem retro — May 4 Production DB Incident\n"
             "Monday May 6, 2026  |  10:00–11:00 AM PDT\n"
             "Zoom: https://meet.company.com/postmortem-042\n"
             "Organizer: alice@company.com\n\n"
             "Attendees:\n"
             "  alice   ✓ accepted\n"
             "  bob     ✓ accepted\n"
             "  carol   ✓ accepted\n"
             "  me      — awaiting response\n"
             "  dave    — awaiting response\n\n"
             "Agenda:\n"
             "  1. Timeline walkthrough (10 min)\n"
             "  2. Root cause analysis (15 min)\n"
             "  3. Action items review (20 min)\n"
             "  4. Prevention measures (15 min)\n\n"
             "Pre-read: https://docs.company.com/postmortem-042\n\n"
             "[ Accept ]  [ Decline ]  [ Maybe ]\n\n"
             "— Google Calendar"
         ),
         is_read=False, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T09:00:00Z", has_attachment=False),

    # ── THREAD t018: Stripe receipt ────────────────────────────────────────────
    dict(id="e029", thread_id="t018", folder="inbox",
         from_addr="receipts@stripe.com", to_addr="me@company.com",
         cc="",
         subject="Receipt from JetBrains — $149.00",
         body=(
             "Thank you for your purchase.\n\n"
             "Date:    May 1, 2026\n"
             "Receipt: RCPT-2026-04-881\n\n"
             "  JetBrains All Products Pack — Annual  1 × $149.00\n"
             "  Total: $149.00 USD\n\n"
             "Payment: Visa ending 4242\n"
             "PDF: https://dashboard.stripe.com/receipts/rcpt-2026-04-881\n\n"
             "— Stripe"
         ),
         is_read=True, is_starred=False, labels='["receipts"]',
         timestamp="2026-05-01T10:00:00Z", has_attachment=True),

    # ── ARCHIVE ────────────────────────────────────────────────────────────────
    dict(id="e030", thread_id="t019", folder="archive",
         from_addr="carol@company.com", to_addr="me@company.com",
         cc="",
         subject="Welcome to Acme — everything you need for Day 1",
         body=(
             "Welcome to Acme Corp!\n\n"
             "DAY 1 LOGISTICS:\n"
             "  • Arrive 9:00 AM — 100 Market St, Suite 800, SF\n"
             "  • Bring laptop + government ID\n"
             "  • Bob meets you at reception\n\n"
             "ACCOUNTS READY:\n"
             "  • Email, Slack, GitHub, Jira, Notion, 1Password\n"
             "  • AWS read access (write after security training)\n\n"
             "FIRST WEEK SCHEDULE:\n"
             "  Mon 09:00  IT setup + badge\n"
             "  Mon 10:00  1:1 with Alice\n"
             "  Mon 14:00  Engineering team intro\n"
             "  Tue 10:00  Product + design walkthrough\n"
             "  Thu 14:00  First sprint planning\n\n"
             "Engineering wiki: https://notion.so/company/engineering\n\n"
             "Carol\nPeople Operations"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-04-15T10:00:00Z", has_attachment=True),

    dict(id="e031", thread_id="t020", folder="archive",
         from_addr="support@aws.com", to_addr="me@company.com",
         cc="",
         subject="Support case #902833 resolved — EC2 unreachable ap-southeast-1",
         body=(
             "Your support case #902833 has been resolved.\n\n"
             "Issue:     EC2 i-0abc123 unreachable in ap-southeast-1a\n"
             "Cause:     AZ-level networking failure (internal AWS component)\n"
             "Resolved:  14:32 UTC  |  Impact duration: ~2h 15m\n\n"
             "Recommendation: Distribute production instances across multiple AZs "
             "using an Auto Scaling Group.\n\n"
             "— AWS Support"
         ),
         is_read=True, is_starred=False, labels='[]',
         timestamp="2026-04-20T15:00:00Z", has_attachment=False),

    dict(id="e032", thread_id="t021", folder="archive",
         from_addr="hr@company.com", to_addr="me@company.com",
         cc="",
         subject="Benefits enrollment — deadline April 30",
         body=(
             "Benefits enrollment closes April 30. Not enrolling = auto-enrolled in defaults.\n\n"
             "Health:  Kaiser HMO (default) | Blue Shield PPO | HDHP + HSA\n"
             "Dental:  Delta Basic (default) | Delta Plus\n"
             "Vision:  VSP Basic (default)\n"
             "401k:    3% match default — recommend at least 6%\n\n"
             "Portal: https://benefits.company.com\n\n— HR"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-04-25T09:00:00Z", has_attachment=False),

    dict(id="e033", thread_id="t022", folder="archive",
         from_addr="noreply@github.com", to_addr="me@company.com",
         cc="",
         subject="[forge-rl] Your PR #309 was merged",
         body=(
             "Congratulations! alice merged PR #309:\n"
             "Fix CORS headers for embedded environments\n\n"
             "https://github.com/company/forge-rl/pull/309\n\n— GitHub"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-02T18:01:00Z", has_attachment=False),

    # ── SENT ───────────────────────────────────────────────────────────────────
    dict(id="e040", thread_id="t001", folder="sent",
         from_addr="me@company.com", to_addr="carol@company.com",
         cc="engineering@company.com",
         subject="Re: [ALERT] P0 — joining bridge now",
         body=(
             "On it. Joining now. Pulling slow query logs:\n\n"
             "  SELECT query, mean_exec_time, calls\n"
             "  FROM pg_stat_statements\n"
             "  ORDER BY mean_exec_time DESC\n"
             "  LIMIT 20;\n\n"
             "billing_queries.get_user_invoices at 7.2s avg. "
             "Users table JOIN — confirms Bob's theory."
         ),
         is_read=True, is_starred=False, labels='["work","urgent"]',
         timestamp="2026-05-04T02:22:00Z", has_attachment=False),

    dict(id="e041", thread_id="t004", folder="sent",
         from_addr="me@company.com", to_addr="alice@company.com",
         cc="",
         subject="Re: 1:1 Thursday — agenda confirmed",
         body=(
             "Thursday 2pm works. I'd like to add:\n"
             "  • Async worker tech design — have a draft, want your feedback\n"
             "  • On-call rotation docs feedback\n"
             "  • Quick question on the promo packet format\n\nSee you then!"
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-06T15:00:00Z", has_attachment=False),

    dict(id="e042", thread_id="t011", folder="sent",
         from_addr="me@company.com", to_addr="bob@company.com",
         cc="",
         subject="Re: Sprint 24 — yes to pairing on FORGE-371",
         body=(
             "Yeah, 27pts + on-call is a lot. FORGE-371 is definitely more than 8 pts.\n\n"
             "Yes to pairing — have questions on v1 backwards compat. DM me a time."
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-05T16:00:00Z", has_attachment=False),

    dict(id="e043", thread_id="t010", folder="sent",
         from_addr="me@company.com", to_addr="bob@company.com",
         cc="",
         subject="Re: Lunch Tuesday — I'm in",
         body="Yes! 12:30 Tuesday works. Ramen + sprint sync combo. I'll bring the Jira board 😅",
         is_read=True, is_starred=False, labels='["personal"]',
         timestamp="2026-05-01T12:00:00Z", has_attachment=False),

    dict(id="e044", thread_id="t015", folder="sent",
         from_addr="me@company.com", to_addr="noreply@github.com",
         cc="",
         subject="Re: Issue #321 — will fix today",
         body=(
             "Good catch Eve — real bug. /forge/reset needs to also clear action_log.\n\n"
             "Hotfix PR today. 5-line change."
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T12:00:00Z", has_attachment=False),

    # ── DRAFTS ─────────────────────────────────────────────────────────────────
    dict(id="e050", thread_id=None, folder="drafts",
         from_addr="me@company.com", to_addr="carol@company.com",
         cc="",
         subject="PTO Request — June 16–27",
         body=(
             "Hi Carol,\n\n"
             "Requesting PTO June 16–27 (2 weeks).\n\n"
             "Coverage arranged:\n"
             "  • Week 1 (Jun 16-20): Bob covers on-call\n"
             "  • Week 2 (Jun 23-27): Alice handles critical issues\n"
             "  • FORGE-421 and FORGE-371 completed before I leave\n\n"
             "Let me know if this works!\n\nThanks,"
         ),
         is_read=True, is_starred=False, labels='[]',
         timestamp="2026-05-03T20:00:00Z", has_attachment=False),

    dict(id="e051", thread_id="t014", folder="drafts",
         from_addr="me@company.com", to_addr="dave@company.com",
         cc="",
         subject="Re: Figma comment — event timeline data model",
         body=(
             "Hey Dave,\n\n"
             "Good question. Currently there's no 24h limit enforced in the API — "
             "that's actually the bug FORGE-421 is fixing.\n\n"
             "For the timeline: treat 24h as a display cap for now. "
             "The API limit will land with FORGE-421.\n\n"
             "Happy to jump on a call if helpful."
         ),
         is_read=True, is_starred=False, labels='["work"]',
         timestamp="2026-05-04T14:00:00Z", has_attachment=False),

    # ── TRASH ─────────────────────────────────────────────────────────────────
    dict(id="e060", thread_id=None, folder="trash",
         from_addr="noreply@coursera.org", to_addr="me@company.com",
         cc="",
         subject="Last chance: 70% off ML Specialization this weekend",
         body="FLASH SALE — 70% off. Code: WEEKEND70. Expires Sunday midnight.\nhttps://coursera.org/ml\n\nUnsubscribe",
         is_read=True, is_starred=False, labels='[]',
         timestamp="2026-05-01T07:00:00Z", has_attachment=False),

    dict(id="e061", thread_id=None, folder="trash",
         from_addr="deals@uber.com", to_addr="me@company.com",
         cc="",
         subject="$10 Uber Cash expiring in 3 days",
         body="Your $10 Uber Cash expires May 7. Use it on your next ride or Uber Eats order.\n\nUnsubscribe",
         is_read=True, is_starred=False, labels='[]',
         timestamp="2026-05-04T10:00:00Z", has_attachment=False),
]

_SEED_CONTACTS = [
    dict(email="alice@company.com",                   name="Alice Chen",       avatar_initial="A"),
    dict(email="bob@company.com",                     name="Bob Martinez",     avatar_initial="B"),
    dict(email="carol@company.com",                   name="Carol Smith",      avatar_initial="C"),
    dict(email="dave@company.com",                    name="Dave Kim",         avatar_initial="D"),
    dict(email="eve@company.com",                     name="Eve Okafor",       avatar_initial="E"),
    dict(email="noreply@github.com",                  name="GitHub",           avatar_initial="G"),
    dict(email="digest@tldr.tech",                    name="TLDR",             avatar_initial="T"),
    dict(email="billing-alerts@aws.amazon.com",       name="AWS Billing",      avatar_initial="A"),
    dict(email="support@aws.com",                     name="AWS Support",      avatar_initial="A"),
    dict(email="jira@company.atlassian.net",          name="Jira",             avatar_initial="J"),
    dict(email="no-reply@zoom.us",                    name="Zoom",             avatar_initial="Z"),
    dict(email="messages-noreply@linkedin.com",       name="LinkedIn",         avatar_initial="L"),
    dict(email="notify@notion.so",                    name="Notion",           avatar_initial="N"),
    dict(email="noreply@figma.com",                   name="Figma",            avatar_initial="F"),
    dict(email="pagerduty@pagerduty.com",             name="PagerDuty",        avatar_initial="P"),
    dict(email="receipts@stripe.com",                 name="Stripe",           avatar_initial="S"),
    dict(email="calendar-notification@google.com",    name="Google Calendar",  avatar_initial="G"),
    dict(email="hr@company.com",                      name="HR",               avatar_initial="H"),
    dict(email="me@company.com",                      name="Me",               avatar_initial="M"),
]

_SEED_LABELS = [
    dict(id="lb01", name="work",       color="#1a73e8"),
    dict(id="lb02", name="urgent",     color="#d93025"),
    dict(id="lb03", name="personal",   color="#188038"),
    dict(id="lb04", name="newsletter", color="#e37400"),
    dict(id="lb05", name="receipts",   color="#8e24aa"),
]


_AUTO_REPLY_MAP = {
    "t001": {
        "from_addr": "carol@company.com",
        "subject": "Re: [P0] Production DB latency spike — 10x normal",
        "body": "Great work on the investigation! Latency is fully recovered. "
                "I'll write up the postmortem and share for review tomorrow. "
                "Thanks for jumping on this so quickly.",
    },
    "t002": {
        "from_addr": "noreply@github.com",
        "subject": "Re: [forge-rl] PR #312: Add SQLAlchemy persistence layer",
        "body": "alice approved your review and the PR has been merged into main.\n\n"
                "All CI checks passed. Deploy scheduled for tonight at 11pm UTC.\n\n— GitHub",
    },
    "t004": {
        "from_addr": "alice@company.com",
        "subject": "Re: 1:1 sync this week — agenda?",
        "body": "Perfect! Thursday 2pm confirmed. I'll send a calendar invite with the updated agenda.\n\n"
                "The async worker proposal sounds interesting — looking forward to hearing about it.\n\nAlice",
    },
    "t005": {
        "from_addr": "support@aws.com",
        "subject": "Re: AWS Bill Alert: Estimated charges $4,231.00",
        "body": "Thank you for reaching out. We've reviewed your account and identified "
                "two EC2 instances running in us-east-1 that may have been left running unintentionally.\n\n"
                "You can review and stop them at: https://console.aws.amazon.com/ec2\n\n— AWS Support",
    },
    "t012": {
        "from_addr": "alice@company.com",
        "subject": "Re: Promo packet — deadline extended to May 20",
        "body": "Great to hear you're working on it! I've blocked time Thursday afternoon "
                "to review your draft. Just share the doc whenever you're ready.\n\nAlice",
    },
}


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
        db.query(Email).delete()
        db.query(Contact).delete()
        db.query(Label).delete()
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
        email_id = f"i{uuid.uuid4().hex[:8]}"
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
            timestamp=_now(),
            has_attachment=False,
        ))
        _log_action(db, "receive", email_id, {"from": req.from_addr, "subject": req.subject})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "received", "email_id": email_id, "state": state}


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
        _log_action(db, "compose", draft_id)
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
        email.timestamp = _now()
        _log_action(db, "send", req.draft_id, {"to": to_addr})
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
        thread = original.thread_id or original.id
        orig_from = original.from_addr
        _log_action(db, "reply", reply_id, {"to": orig_from})
        db.commit()
        # Auto-inject simulated response if configured and not already done
        if thread in _AUTO_REPLY_MAP:
            already = db.query(ActionLog).filter(
                ActionLog.action_type == "auto_reply",
                ActionLog.target_id == thread,
            ).first()
            if not already:
                r = _AUTO_REPLY_MAP[thread]
                resp_id = f"ar{uuid.uuid4().hex[:8]}"
                db.add(Email(
                    id=resp_id, thread_id=thread, folder="inbox",
                    from_addr=r["from_addr"], to_addr="me@company.com", cc="",
                    subject=r["subject"], body=r["body"],
                    snippet=_snippet(r["body"]),
                    is_read=False, is_starred=False, labels='["work"]',
                    timestamp=_now(), has_attachment=False,
                ))
                _log_action(db, "auto_reply", thread, {"from": r["from_addr"]})
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
        _log_action(db, "forward", fwd_id, {"to": req.to})
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
        _log_action(db, "archive", req.email_id)
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
        _log_action(db, "delete", req.email_id, {"subject": subject, "was_starred": was_starred})
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
        _log_action(db, "mark_read", req.email_id, {"read": req.read})
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
        _log_action(db, "star", req.email_id, {"starred": req.starred})
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
        _log_action(db, "label", req.email_id)
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
        _log_action(db, "move", req.email_id, {"folder": req.folder})
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
        _log_action(db, "bulk_archive", None, {"count": len(archived)})
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
        _log_action(db, "create_label", None, {"name": req.name})
        db.commit()
        state = _get_state_dict(db)
    return {"status": "created", "label_id": lb_id, "name": req.name, "state": state}


@app.post("/empty_trash")
def empty_trash():
    with SessionLocal() as db:
        count = db.query(Email).filter(Email.folder == "trash").count()
        db.query(Email).filter(Email.folder == "trash").delete()
        _log_action(db, "empty_trash", None, {"deleted_count": count})
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
