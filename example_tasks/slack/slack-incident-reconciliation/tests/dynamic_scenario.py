"""Shared deterministic driver for dynamic-scenario self-tests."""

from __future__ import annotations

from pathlib import Path

from slack_sim.identity import LOGGED_IN_USER
from slack_sim.service import execute_tool, export_state, seed_database

ACTOR = LOGGED_IN_USER.user_id


def seeded(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    db = root / "slack.db"
    seed_database(db, root / "seed.sql")
    return db


def call(db: Path, tool: str, payload: dict) -> dict:
    return execute_tool(db, tool, payload, ACTOR)


def drive_terminal(db: Path, *, answer: bool = True) -> dict:
    call(db, "search_messages", {"query": "IDP-ACME-014"})
    call(db, "search_messages", {"query": "ACME-ACCESS-04"})
    call(db, "search_messages", {"query": "DIR-PAGE-311"})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "The implementation uses includes/substring matching, but the SAML audience requirement is exact equality. A containing malicious audience would pass; use === after canonicalization."})
    call(db, "get_thread_replies", {"channel_id": "C023", "thread_ts": _ts(db, "MSG194")})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "Looks good to me"})
    call(db, "get_channel_messages", {"channel_id": "C020"})
    call(db, "get_channel_messages", {"channel_id": "C020"})

    call(db, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})
    call(db, "search_messages", {"query": "26 synthetic", "limit": 50})
    call(db, "get_thread_replies", {"channel_id": "C023", "thread_ts": _ts(db, "MSG200")})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})

    call(db, "reply_to_thread", {"thread_parent_id": "MSG214", "body": "Using some gives any-match semantics, but every required entitlement must be present. A user with only one group passes incorrectly; use every/all."})
    call(db, "get_thread_replies", {"channel_id": "C023", "thread_ts": _ts(db, "MSG214")})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG214", "body": "Looks good to me"})
    call(db, "get_channel_messages", {"channel_id": "C022"})
    call(db, "get_channel_messages", {"channel_id": "C022"})
    call(db, "get_channel_messages", {"channel_id": "C022"})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG219", "body": "Looks good to me"})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG224", "body": "The DirectoryUsers API may repeat a boundary user while the collection changes, and provisionUser is non-idempotent, so this can provision one user twice. Deduplicate stable IDs across pages or use a snapshot."})

    call(db, "reply_to_thread", {"thread_parent_id": "CUT001", "body": "Nina, can you confirm whether 9:30 was an actual approved change request or only a question?"})
    call(db, "get_channel_messages", {"channel_id": "C019"})
    # Cutover coordination moved to its own channel, and Ben is not on it.
    call(db, "search_channels", {"query": "cutover"})
    call(db, "join_channel", {"channel_id": "C024"})
    call(db, "get_channel_messages", {"channel_id": "C024"})
    call(db, "get_thread_replies", {"channel_id": "C024", "thread_ts": _ts(db, "CUT002")})
    call(db, "reply_to_thread", {"thread_parent_id": "LAT022", "body": "Sam, please verify the rollback worker by running the recovery invocation and report its status."})
    rehearsal_page = call(db, "get_channel_messages", {"channel_id": "C024"})
    rehearsal_root = next(message for message in rehearsal_page["messages"] if message["id"] == "LAT027")
    call(db, "get_thread_replies", {"channel_id": "C024", "thread_ts": rehearsal_root["thread_ts"]})
    call(db, "reply_to_thread", {"thread_parent_id": "LAT027", "body": "The rollback rehearsal exposed a stale recovery config pin. Correct the pin and rerun the rollback invocation before closing rehearsal."})
    call(db, "get_thread_replies", {"channel_id": "C024", "thread_ts": rehearsal_root["thread_ts"]})

    if answer:
        call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "BLOCKED. Nina confirmed 9:30 was only a question; Thursday August 20, 2026 remains 9:00 PM PT, 90 minutes, about 15 minutes downtime (LAT021). SSO owner Priya Shah: audience matching is fixed, but EU-2 remains on the acme-eu-2025 regional signature/key metadata failure (LAT019). Historical export owner Ahmed Khan: VERIFIED COMPLETE after March reconciled with 26 exclusions (LAT010). EU permissions owner Marcus Reed: VERIFIED COMPLETE after both restricted workflows passed (LAT020). Sam Okafor corrected the stale rollback config pin and the invocation passed (LAT033). Daniel's component rehearsal is complete with data, EU access, rollback, and monitoring verified while auth reproduced the known blocker (LAT035). Tessa has identity bridge coverage while Priya owns SSO; Ahmed data, Marcus workspace, Daniel coordination, Nina customer contact (LAT023). Only SSO remains blocking."})
    return export_state(db)


def _ts(db: Path, message_id: str) -> str:
    return next(message["ts"] for message in export_state(db)["messages"] if message["message_id"] == message_id)


