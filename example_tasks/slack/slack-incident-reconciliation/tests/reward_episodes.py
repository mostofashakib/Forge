"""Two realistic episodes, for checking that the reward means what it says.

Neither is the oracle. `competent_run` is a second, independently written
solution: a different order of work, evidence reached by different routes, the
timing question asked by DM rather than in-thread, and a final update split
across two messages. If the reward is a statement about the work rather than
about one script, it has to pay this run in full too.

`capability_failure_run` is modelled on what the recorded Opus 4.7 episodes
actually did: read the obvious channels, approve the reviews that look fine,
never discover that coordination moved to a channel it was not in, and answer
Daniel from the state it had. Nothing dishonest happens in it -- which is the
point. A fair reward has to separate "did less of the work" from "cheated".
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

from slack_sim.service import execute_tool, export_state, seed_database

ACTOR = "U002"
NINA, SAM = "U046", "U019"
ACME, IDENTITY, DATA, SUPPORT, DEBUG, BRIDGE = "C019", "C020", "C021", "C022", "C023", "C024"


def seeded(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    db = root / "slack.db"
    seed_database(db, root / "seed.sql")
    return db


def call(db: Path, tool: str, payload: dict) -> dict:
    return execute_tool(db, tool, payload, ACTOR)


def _ts(db: Path, message_id: str) -> str:
    return next(m["ts"] for m in export_state(db)["messages"] if m["message_id"] == message_id)


def _safe(db: Path, tool: str, payload: dict) -> dict | None:
    """Run a call, or report that it was refused.

    The steps below are also used one-at-a-time-removed, to price what each is
    worth. Leaving one out can make a later call impossible -- reading the
    bridge without having joined it, replying in a thread that was never
    published -- and that is the realistic consequence, not a crash. An agent
    that skipped the join would not have been reading the bridge either.
    """
    try:
        return call(db, tool, payload)
    except Exception:  # noqa: BLE001 - a refused call is an outcome, not a fault
        return None


def _orient(db: Path) -> None:
    call(db, "list_channels", {})
    call(db, "get_channel_messages", {"channel_id": ACME})
    call(db, "get_thread_replies", {"channel_id": ACME, "thread_ts": _ts(db, "MSG145")})


def _review_permissions_critique(db: Path) -> None:
    _safe(db, "get_thread_replies", {"channel_id": DEBUG, "thread_ts": _ts(db, "MSG214")})
    _safe(db, "reply_to_thread", {"thread_parent_id": "MSG214", "body": "some() accepts any single match, but every required entitlement has to be present. A user holding one group would pass; switch to every()."})


def _review_permissions_approve(db: Path) -> None:
    _safe(db, "get_thread_replies", {"channel_id": DEBUG, "thread_ts": _ts(db, "MSG214")})
    _safe(db, "reply_to_thread", {"thread_parent_id": "MSG214", "body": "Looks good to me"})


def _review_audience_critique(db: Path) -> None:
    _safe(db, "get_thread_replies", {"channel_id": DEBUG, "thread_ts": _ts(db, "MSG194")})
    _safe(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "Audience validation uses substring containment where the spec requires exact equality, so a longer hostile audience would be accepted. Compare with === after canonicalization."})


def _review_audience_approve(db: Path) -> None:
    _safe(db, "get_thread_replies", {"channel_id": DEBUG, "thread_ts": _ts(db, "MSG194")})
    _safe(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "Looks good to me"})


def _review_retry_approve(db: Path) -> None:
    _safe(db, "reply_to_thread", {"thread_parent_id": "MSG219", "body": "Looks good to me"})


def _review_export_approve(db: Path) -> None:
    _safe(db, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})


def _review_paging_critique(db: Path) -> None:
    _safe(db, "get_thread_replies", {"channel_id": DEBUG, "thread_ts": _ts(db, "MSG224")})
    _safe(db, "reply_to_thread", {"thread_parent_id": "MSG224", "body": "Paging over DirectoryUsers can repeat a boundary user while the collection shifts, and provisionUser is not idempotent, so someone gets provisioned twice. Deduplicate on stable ids or read from a snapshot."})


def _search_deep_history(db: Path) -> None:
    """The prerequisite discussions, reached from the identifiers the threads cite."""
    for query in ("IDP-ACME-014", "ACME-ACCESS-04", "DIR-PAGE-311"):
        call(db, "search_messages", {"query": query})
    call(db, "search_messages", {"query": "26 synthetic", "limit": 50})


def _read_working_channels(db: Path) -> None:
    call(db, "get_channel_messages", {"channel_id": IDENTITY})
    call(db, "get_channel_messages", {"channel_id": IDENTITY})
    call(db, "get_channel_messages", {"channel_id": DATA})
    for _ in range(3):
        call(db, "get_channel_messages", {"channel_id": SUPPORT})


def _reopened_export_approve(db: Path) -> None:
    """The backfill result reopens the export review, so it gets read again."""
    _safe(db, "get_thread_replies", {"channel_id": DEBUG, "thread_ts": _ts(db, "MSG200")})
    _safe(db, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})


def _ask_about_timing(db: Path) -> None:
    """Asked of the person who can settle it -- by DM, because she raised 9:30."""
    call(db, "send_dm_message", {"recipient_id": NINA, "body": "Nina, was 9:30 ever an approved change, or was it only a question? I need the confirmed start time for Thursday."})


def _find_and_join_the_bridge(db: Path) -> None:
    """Coordination moved. Find it, join it, read it."""
    call(db, "search_channels", {"query": "cutover"})
    call(db, "join_channel", {"channel_id": BRIDGE})
    _safe(db, "get_channel_messages", {"channel_id": BRIDGE})
    _safe(db, "get_thread_replies", {"channel_id": BRIDGE, "thread_ts": _ts(db, "CUT002")})


def _ask_about_rollback(db: Path) -> None:
    call(db, "send_dm_message", {"recipient_id": SAM, "body": "Sam, can you run the rollback recovery invocation and confirm its verification status? Rollback is on the critical path and I have no evidence it was tested."})


def _close_the_rehearsal(db: Path) -> None:
    """The rehearsal reopens what rollback had closed."""
    page = _safe(db, "get_channel_messages", {"channel_id": BRIDGE})
    root = next((m for m in (page or {}).get("messages", []) if m["id"] == "LAT027"), None)
    if root is None:
        return
    call(db, "get_thread_replies", {"channel_id": BRIDGE, "thread_ts": root["thread_ts"]})
    call(db, "reply_to_thread", {"thread_parent_id": "LAT027", "body": "The rehearsal caught a stale recovery config pin. Correct the pin and rerun the rollback invocation before we close rehearsal."})
    call(db, "get_thread_replies", {"channel_id": BRIDGE, "thread_ts": root["thread_ts"]})
    call(db, "list_notifications", {})


def _final_report(db: Path) -> None:
    """The update, split the way a long one gets split."""
    call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "Timing first: Nina confirmed 9:30 was only a question, so Thursday August 20 2026 stands at 9:00 PM PT, a 90 minute window, about 15 minutes of customer-visible downtime (LAT021). Authentication is owned by Priya Shah and is still failing: the audience defect is fixed and deployed, but EU-2 fails signature verification because its regional metadata is missing acme-eu-2025 (LAT019). Historical export, owner Ahmed Khan, is verified complete -- March reconciled with 26 documented exclusions (LAT010). EU permissions, owner Marcus Reed, verified complete after both restricted workflows passed with the customer (LAT020)."})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "Rollback: Sam Okafor corrected the stale recovery config pin and the rerun passed (LAT033). Rehearsal, owner Daniel Cho, is complete -- data, EU access, rollback and monitoring all verified, with authentication reproducing the known blocker (LAT035). Bridge coverage, owner Daniel Cho: Tessa holds identity while Priya owns SSO, Ahmed data, Marcus workspace, Nina customer contact (LAT023). Verdict: BLOCKED. SSO is the only outstanding blocker and it needs Priya to fix the EU-2 key metadata path and rerun the full smoke test before we can call this ready."})


#: The competent run, in the order it happens, with each piece of work named.
#: Naming them is what lets the score table price them one at a time.
STEPS: tuple[tuple[str, Any], ...] = (
    ("orient", _orient),
    ("review_permissions_critique", _review_permissions_critique),
    ("review_permissions_approve", _review_permissions_approve),
    ("review_audience_critique", _review_audience_critique),
    ("review_audience_approve", _review_audience_approve),
    ("review_retry_approve", _review_retry_approve),
    ("review_export_approve", _review_export_approve),
    ("review_paging_critique", _review_paging_critique),
    ("search_deep_history", _search_deep_history),
    ("read_working_channels", _read_working_channels),
    ("reopened_export_approve", _reopened_export_approve),
    ("ask_about_timing", _ask_about_timing),
    ("find_and_join_the_bridge", _find_and_join_the_bridge),
    ("ask_about_rollback", _ask_about_rollback),
    ("close_the_rehearsal", _close_the_rehearsal),
    ("final_report", _final_report),
)

STEP_NAMES = tuple(name for name, _ in STEPS)


def competent_run(db: Path, skip: Collection[str] = ()) -> dict:
    """A strong agent working the problem in its own order.

    `skip` omits named steps, which is how the score table asks what each piece
    of the work is worth: run everything but one thing, and see what the reward
    loses.
    """
    skipped = set(skip)
    unknown = skipped - set(STEP_NAMES)
    if unknown:
        raise KeyError(f"no such step: {sorted(unknown)}")
    for name, step in STEPS:
        if name not in skipped:
            step(db)
    return export_state(db)


def capability_failure_run(db: Path) -> dict:
    """An able agent that stopped short. Modelled on the recorded runs."""
    call(db, "list_channels", {})
    call(db, "get_channel_messages", {"channel_id": ACME})
    call(db, "get_thread_replies", {"channel_id": ACME, "thread_ts": _ts(db, "MSG145")})
    call(db, "get_channel_messages", {"channel_id": DEBUG})
    call(db, "get_channel_messages", {"channel_id": IDENTITY})
    call(db, "search_messages", {"query": "SSO"})

    # The reviews that look fine on their face get approved; the ones needing a
    # read of the implementation get approved too.
    call(db, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG219", "body": "Looks good to me"})
    call(db, "get_thread_replies", {"channel_id": DEBUG, "thread_ts": _ts(db, "MSG194")})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "The audience check should compare exactly rather than by substring; a longer audience containing ours would be accepted."})

    # It never learns that coordination moved, so it never asks anyone anything.
    call(db, "get_channel_messages", {"channel_id": SUPPORT})
    call(db, "list_notifications", {})

    call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "Status as far as I can tell: BLOCKED. SSO is still failing on the EU side and Priya Shah owns it. The historical export backfill looks complete and Ahmed Khan owns it. EU permissions are still being checked. I could not confirm the migration window or whether rollback has been tested."})
    return export_state(db)
