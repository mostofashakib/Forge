"""Verifier-only truth for the dynamic Acme migration episode."""

from __future__ import annotations

import re
from typing import Any

from slack_sim.seed import (
    BENCHMARK_NOW, SCENARIO_EVENT_IDS, SLACK_LATENT_MESSAGES, SLACK_MESSAGES, SLACK_USERS,
)

REQUEST_MESSAGE_ID = "MSG145"

GROUND_TRUTH: dict[str, Any] = {
    "overall_readiness": "BLOCKED",
    "migration": {"date": "2026-08-20", "start_time": "21:00", "timezone": "America/Los_Angeles", "duration_minutes": 90, "expected_downtime_minutes": 15, "authoritative_evidence_message_id": "LAT021"},
    "sso": {"status": "BLOCKED", "owner": "Priya Shah", "owner_user_id": "U042", "reason": "EU-2 fails signature/key verification because its regional metadata path lacks acme-eu-2025", "next_action": "Correct the EU-2 metadata/key path and successfully rerun the full smoke test", "authoritative_evidence_message_id": "LAT019"},
    "historical_export": {"status": "VERIFIED COMPLETE", "owner": "Ahmed Khan", "owner_user_id": "U045", "reason": "March reconciles after exactly 26 documented synthetic exclusions", "next_action": "None", "authoritative_evidence_message_id": "LAT010"},
    "eu_permissions": {"status": "VERIFIED COMPLETE", "owner": "Marcus Reed", "owner_user_id": "U044", "reason": "EU-Finance and EU-Legal completed the required restricted workflow", "next_action": "None", "authoritative_evidence_message_id": "LAT020"},
    "rollback": {"status": "VERIFIED READY", "owner": "Sam Okafor", "owner_user_id": "U019", "reason": "The stale recovery-config pin found by rehearsal was corrected and rollback invocation passed", "next_action": "None", "authoritative_evidence_message_id": "LAT033"},
    "rehearsal": {"status": "COMPLETE", "owner": "Daniel Cho", "owner_user_id": "U041", "reason": "Data, EU access, corrected rollback, and monitoring passed while authentication reproduced the tracked EU-2 blocker", "next_action": "Carry the SSO blocker into go/no-go", "authoritative_evidence_message_id": "LAT035"},
    "bridge_coverage": {"status": "COVERED", "owner": "Daniel Cho", "owner_user_id": "U041", "reason": "Tessa covers identity bridge while Priya owns SSO; Ahmed data; Marcus workspace; Nina customer contact", "next_action": "Maintain coverage through cutover", "authoritative_evidence_message_id": "LAT023"},
}


CODE_REVIEWS: dict[str, dict[str, Any]] = {
    "normalizeCursor": {"thread": "MSG186", "open": False, "expected": "LGTM"},
    "audienceMatches": {"thread": "MSG194", "open": True, "expected": "CRITIQUE", "keywords": (frozenset({"includes", "substring", "contains", "partial", "prefix"}), frozenset({"exact", "equality", "===", "verbatim", "identical"}))},
    "buildPartitionWindow": {"thread": "MSG200", "open": True, "expected": "LGTM"},
    "flushBatch": {"thread": "MSG207", "open": False, "expected": "LGTM"},
    "hasRequiredAccess": {"thread": "MSG214", "open": True, "expected": "CRITIQUE", "keywords": (frozenset({"some", "any", "one", "single", "partial"}), frozenset({"every", "all", "each", "both"}))},
    "Retry": {"thread": "MSG219", "open": True, "expected": "LGTM"},
    "provisionAllUsers": {"thread": "MSG224", "open": True, "expected": "CRITIQUE", "keywords": (frozenset({"duplicate", "repeat", "boundary", "again", "re-run", "rerun", "retry"}), frozenset({"non-idempotent", "not idempotent", "twice", "duplicate side effect", "provisioned twice", "duplicates"}))},
}
OPEN_REVIEWS = tuple(name for name, review in CODE_REVIEWS.items() if review["open"])
CLOSED_REVIEWS = tuple(name for name, review in CODE_REVIEWS.items() if not review["open"])

REQUIRED_EVENTS = frozenset(SCENARIO_EVENT_IDS)
STALE_CLAIMS = frozenset({"MSG125", "MSG128", "MSG135", "MSG137", "MSG138", "MSG139", "MSG140", "MSG141", "MSG143", "MSG146", "MSG164", "MSG157", "MSG169", "MSG181", "LAT015", "LAT025", "LAT026", "LAT031"})
#: The message that settles each area, by area. The same fact `GROUND_TRUTH`
#: carries, named here so the supersession map below can be checked against it.
AUTHORITATIVE_BY_AREA: dict[str, str] = {
    area: str(truth["authoritative_evidence_message_id"])
    for area, truth in GROUND_TRUTH.items()
    if isinstance(truth, dict) and "authoritative_evidence_message_id" in truth
}

#: What later contradicted each stale claim -- the scenario's own spine, written
#: down so a grader can use it.
#:
#: This replaced a list of dismissal words. Whether a report *rests on* a
#: superseded claim or *names it in order to reject it* is a question about
#: language, and matching words answered it wrongly in both directions: it
#: charged reports that rejected a claim in their own phrasing, and it let
#: through a report that wrote "superseded" beside a claim and then relied on
#: it. Asking instead whether the newer evidence is cited needs no reading at
#: all, and cannot be satisfied by vocabulary.
#:
#: A claim that speaks to several areas lists all of them. It is charged only
#: when none of its successors is cited: under-charging is the right direction
#: for an error, and the coverage checks price the missing evidence anyway.
SUPERSEDED_BY: dict[str, tuple[str, ...]] = {
    "MSG125": ("LAT020",),                     # kickoff ownership: Lena -> Marcus
    "MSG128": ("LAT019",),                     # "SSO complete"
    "MSG135": ("LAT021",),                     # "Thursday 9:00 PM confirmed"
    "MSG137": ("LAT010",),                     # "data export done"
    "MSG138": ("LAT020",),                     # "permissions updated"
    "MSG139": ("LAT021",),                     # "see everyone Thursday at 8"
    "MSG140": ("LAT020", "LAT021"),            # "basically fixed / fine for Thursday"
    "MSG141": ("LAT019", "LAT010", "LAT020"),  # the three-way recap
    "MSG143": ("LAT020",),                     # "permissions should be good now"
    "MSG146": ("LAT019",),                     # "cert rotation complete"
    "MSG157": ("LAT019",),                     # "US passes, EU failing"
    "MSG164": ("LAT010",),                     # "export completed successfully"
    "MSG169": ("LAT010",),                     # "March partition excluded"
    "MSG181": ("LAT020",),                     # "Finance confirmed, Legal pending"
    "LAT015": ("LAT020",),                     # partial permissions verification
    "LAT025": ("LAT033",),                     # dry-run rollback
    "LAT026": ("LAT033",),                     # provisionally verified
    "LAT031": ("LAT033",),                     # rollback FAIL on a stale config pin
}

ISSUES = ("sso", "historical_export", "eu_permissions")
OPERATIONAL_AREAS = ("rollback", "rehearsal", "bridge_coverage")
DEEP_EVIDENCE = {
    "sso": {"root": "MSG306", "channel": "C020"},
    "export": {"root": "MSG551", "channel": "C021"},
    "permissions": {"root": "MSG795", "channel": "C012"},
    "pagination": {"root": "MSG799", "channel": "C012"},
}


# The predicates that say what a *correct* review decision looks like. They
# live beside the answer key rather than inside one grader, because two graders
# now ask the question and a second copy would be free to drift from this one.
LGTM = "looks good to me"


def is_lgtm(body: str) -> bool:
    """An unqualified approval, whatever trailing punctuation it carries."""
    return re.sub(r"[\s.!,]+$", "", body.strip().lower()) == LGTM


def contains_groups(text: str, groups: tuple[frozenset[str], ...]) -> bool:
    """One word from every group -- the defect named in the reviewer's own words.

    Each group is a way of saying one necessary half of the defect, so a review
    must hit all of them. What a group must never be is the *remedy*: the
    pagination review once needed a word from {dedup, seen, snapshot} as well,
    which asked the reviewer to design the fix before their critique counted.
    """
    lowered = text.lower()
    return all(any(word in lowered for word in group) for group in groups)


def critique_ok(body: str, review: dict[str, Any]) -> bool:
    return not is_lgtm(body) and contains_groups(body, review["keywords"])


def after(message: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Strictly later in the workspace's own ordering, ties broken by id."""
    return (message["ts"], message["message_id"]) > (reference["ts"], reference["message_id"])


def validate_against_seed() -> None:
    visible = {message.message_id: message for message in SLACK_MESSAGES}
    latent = {message.message_id: message for message in SLACK_LATENT_MESSAGES}
    if len(visible) < 1000 or len(visible) != len(SLACK_MESSAGES):
        raise ValueError("dynamic benchmark requires at least 1,000 unique visible messages")
    if REQUEST_MESSAGE_ID not in visible:
        raise ValueError("request message is missing")
    for issue in ISSUES + OPERATIONAL_AREAS:
        truth = GROUND_TRUTH[issue]
        if truth["authoritative_evidence_message_id"] not in latent:
            raise ValueError(f"{issue} final evidence must be latent at T0")
        if not any(u.user_id == truth["owner_user_id"] and u.display_name == truth["owner"] for u in SLACK_USERS):
            raise ValueError(f"{issue} owner is not seeded")
    for item in DEEP_EVIDENCE.values():
        root = visible[item["root"]]
        newer_roots = sum(1 for message in SLACK_MESSAGES if message.conversation_id == item["channel"] and message.thread_parent_id is None and message.created_step > root.created_step)
        if newer_roots <= 200:
            raise ValueError(f"{item['root']} is only {newer_roots} top-level messages deep")
    if max(message.created_step for message in SLACK_MESSAGES) >= BENCHMARK_NOW:
        raise ValueError("visible seed contains a future message")


validate_against_seed()
