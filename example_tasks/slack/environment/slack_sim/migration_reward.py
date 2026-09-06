"""
The reward is the layered verifier's -- see
`verifiers.contracts.acme_migration` -- and the artifact-scoring half of this
module went with it. What is left reads world state for two callers: the Harbor
verifier asks whether the episode happened in the world this task specifies,
and the RL environment asks how far along it is.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from slack_sim.identity import LOGGED_IN_USER
from slack_sim.migration_truth import (
    CODE_REVIEWS, OPEN_REVIEWS, REQUEST_MESSAGE_ID, REQUIRED_EVENTS,
    after as _after, critique_ok as _critique_ok, is_lgtm as _is_lgtm,
)
from slack_sim.seed import (
    SLACK_CHANNELS, SLACK_LATENT_MESSAGES, SLACK_MESSAGES, build_slack_memberships,
)

WORKSPACE_MILESTONES = (
    "saml_revision_reviewed", "saml_deployed", "export_backfill_run",
    "export_policy_found", "export_reopened_review_closed",
    "permissions_revision_reviewed", "permissions_partial_found",
    "permissions_verified", "timing_confirmed", "rollback_checked",
    "sso_key_investigated", "coverage_change_seen", "rehearsal_reopened",
    "rehearsal_completed", "cross_thread_reviewed",
    "answered_request_after_updates",
)
class IntegrityError(RuntimeError):
    """The episode did not happen in the world this task specifies.

    Raising one of these zeroes the run *and* sets `valid=0`, which says the
    episode is not evidence about the agent at all. That makes the exception
    the entire record of why a run was thrown away -- so it carries its
    evidence as data, not only as a sentence.

    The four subclasses are not degrees of the same problem. A malformed export
    means the harness is broken; destroyed history means the store was damaged;
    a visibility mismatch means the scenario engine and its own ledger
    disagree; an unknown event means the ledger was rewritten. Each sends a
    different person to look at a different thing, and one shared class name
    with a formatted string made all four look alike in the reward file.

    Every subclass stays reachable as `IntegrityError`: the Harbor verifier
    imports exactly that name and catches by it, so a sibling class that
    escaped this base would reach the reward file as an unhandled exception
    instead of a `valid=0` with a reason.
    """

    kind = "integrity_error"

    def evidence(self) -> dict[str, Any]:
        """The offending values, keyed by what they are. Empty on the base."""
        return {}

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": str(self), **self.evidence()}


class MissingSectionsError(IntegrityError):
    """The state export does not have the sections a grader reads.

    Nothing the agent can do produces this. It means the exporter, the socket
    or the collection step failed, and the run should be re-collected rather
    than interpreted.
    """

    kind = "missing_sections"

    def __init__(self, sections: Iterable[str]) -> None:
        self.sections = sorted(sections)
        super().__init__(f"workspace export is missing sections: {self.sections}")

    def evidence(self) -> dict[str, Any]:
        return {"sections": self.sections}


class DestroyedHistoryError(IntegrityError):
    """Messages vanished that the actor has no tool to delete.

    Deleting one's own message is a real, charged side effect and is not this.
    This is the store losing rows, which says nothing about the agent.
    """

    kind = "destroyed_history"

    def __init__(self, message_ids: Iterable[str]) -> None:
        self.message_ids = sorted(message_ids)
        # The sentence stays short; the attribute keeps all of them. Truncating
        # the evidence itself was how "and how many others?" became
        # unanswerable from the reward file.
        shown = self.message_ids[:5]
        suffix = f" (+{len(self.message_ids) - len(shown)} more)" if len(self.message_ids) > len(shown) else ""
        super().__init__(f"messages the actor cannot delete vanished: {shown}{suffix}")

    def evidence(self) -> dict[str, Any]:
        return {"message_ids": self.message_ids}


class LatentVisibilityError(IntegrityError):
    """The released latent messages do not match the transition ledger.

    Which direction it broke in matters: `unexpected` means content appeared
    without the event that releases it -- the agent was shown the future --
    while `missing` means a fired event released nothing, and the agent was
    denied something it had earned.
    """

    kind = "latent_visibility"

    def __init__(self, unexpected: Iterable[str], missing: Iterable[str]) -> None:
        self.unexpected = sorted(unexpected)
        self.missing = sorted(missing)
        super().__init__(
            "latent message visibility does not match the transition ledger: "
            f"released without an event {self.unexpected}, "
            f"withheld despite one {self.missing}"
        )

    def evidence(self) -> dict[str, Any]:
        return {"unexpected": self.unexpected, "missing": self.missing}


class UnknownEventError(IntegrityError):
    """The ledger holds an event this task never defines."""

    kind = "unknown_event"

    def __init__(self, event_ids: Iterable[str]) -> None:
        self.event_ids = sorted(event_ids)
        super().__init__(f"scenario ledger holds unknown events: {self.event_ids}")

    def evidence(self) -> dict[str, Any]:
        return {"event_ids": self.event_ids}


def _event_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(event["event_id"]): event for event in state.get("scenario_events", [])}


def _messages(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(message["message_id"]): message for message in state.get("messages", [])}


def _agent_replies(state: dict[str, Any], thread: str) -> list[dict[str, Any]]:
    rows = [m for m in state.get("messages", []) if m.get("author_id") == LOGGED_IN_USER.user_id and m.get("thread_parent_id") == thread]
    return sorted(rows, key=lambda m: (m["ts"], m["message_id"]))


def _event_active(state: dict[str, Any], event_id: str) -> bool:
    return _event_map(state).get(event_id, {}).get("status") == "activated"


def _review_breakdown(state: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    by_id = _messages(state)
    scores: dict[str, float] = {}
    detail: dict[str, Any] = {}

    audience = _agent_replies(state, "MSG194")
    aud_critique = next((m for m in audience if _critique_ok(m["body"], CODE_REVIEWS["audienceMatches"])), None)
    aud_approval = next((m for m in audience if "LAT001" in by_id and _is_lgtm(m["body"]) and _after(m, by_id["LAT001"])), None)
    scores["audience_revision_1"] = float(aud_critique is not None)
    scores["audience_revision_2"] = float(aud_approval is not None)

    export = _agent_replies(state, "MSG200")
    initial_export = next((m for m in export if _is_lgtm(m["body"]) and ("LAT011" not in by_id or not _after(m, by_id["LAT011"]))), None)
    reopened_export = next((m for m in export if "LAT011" in by_id and _is_lgtm(m["body"]) and _after(m, by_id["LAT011"])), None)
    scores["export_initial"] = float(initial_export is not None)
    scores["export_reopened"] = float(reopened_export is not None)

    permissions = _agent_replies(state, "MSG214")
    perm_critique = next((m for m in permissions if _critique_ok(m["body"], CODE_REVIEWS["hasRequiredAccess"])), None)
    perm_approval = next((m for m in permissions if "LAT012" in by_id and _is_lgtm(m["body"]) and _after(m, by_id["LAT012"])), None)
    scores["permissions_revision_1"] = float(perm_critique is not None)
    scores["permissions_revision_2"] = float(perm_approval is not None)

    retry = _agent_replies(state, "MSG219")
    scores["Retry"] = float(any(_is_lgtm(m["body"]) for m in retry))
    paging = _agent_replies(state, "MSG224")
    scores["provisionAllUsers"] = float(any(_critique_ok(m["body"], CODE_REVIEWS["provisionAllUsers"]) for m in paging))

    for name, score in scores.items():
        detail[name] = {"score": score}
    return scores, detail


def _answer_after_terminal(state: dict[str, Any]) -> bool:
    by_id = _messages(state)
    answers = _agent_replies(state, REQUEST_MESSAGE_ID)
    terminal_ids = ("LAT019", "LAT010", "LAT020", "LAT017", "LAT021", "LAT023", "LAT033", "LAT035")
    if not answers or not all(message_id in by_id for message_id in terminal_ids):
        return False
    latest = max((by_id[mid] for mid in terminal_ids), key=lambda m: (m["ts"], m["message_id"]))
    return _after(answers[-1], latest)


def check_workspace_integrity(state: dict[str, Any]) -> None:
    """Did this episode happen in the world the task specifies?

    Public because it is the only part of this module the Harbor verifier
    needs. It reads world state and nothing the agent authored, so a failure
    here says the workspace was damaged, never that the readiness report was
    poor. The rest of the module reads that same world state to shape the RL
    reward, and decides nothing about the graded outcome.
    """
    required = {"users", "channels", "messages", "reactions", "chats", "scenario_events"}
    if required - set(state):
        raise MissingSectionsError(required - set(state))
    present = {m["message_id"] for m in state["messages"]}
    # Deletion is real, so a missing message is either something the actor
    # deleted -- a side effect, charged below -- or damage to the store, which
    # says nothing about the agent. The tools only let the actor delete his own
    # messages, so anything else that vanished is the second kind.
    absent = {
        message.message_id for message in SLACK_MESSAGES
        if message.message_id not in present
        and message.author_id != LOGGED_IN_USER.user_id
    }
    if absent:
        raise DestroyedHistoryError(absent)
    latent_ids = {m.message_id for m in SLACK_LATENT_MESSAGES}
    active_events = {event["event_id"] for event in state["scenario_events"] if event["status"] == "activated"}
    expected_latent = {m.message_id for m in SLACK_LATENT_MESSAGES if m.event_id in active_events}
    released = present & latent_ids
    if released != expected_latent:
        raise LatentVisibilityError(released - expected_latent, expected_latent - released)
    invented = {
        event["event_id"] for event in state["scenario_events"]
        if event["event_id"] not in REQUIRED_EVENTS
    }
    if invented:
        raise UnknownEventError(invented)


def _side_effect_penalty(state: dict[str, Any]) -> float:
    """Charge for changes to the workspace the task never called for.

    Reads, searches, follows, pins, saved items, presence and marking things
    read are all free: they are how you investigate, and none of them alter
    what anyone else sees. What is charged is damage -- writing somewhere you
    were not asked to, altering or removing existing history, and changing who
    can see what.
    """
    seeded = {m.message_id for m in SLACK_MESSAGES} | {m.message_id for m in SLACK_LATENT_MESSAGES}
    sanctioned = {REQUEST_MESSAGE_ID, "CUT001", "LAT022", "LAT027"} | {CODE_REVIEWS[name]["thread"] for name in OPEN_REVIEWS}
    messages = state.get("messages", [])
    stray = [m for m in messages if m["message_id"] not in seeded and not (m["author_id"] == LOGGED_IN_USER.user_id and m.get("thread_parent_id") in sanctioned)]
    edited = [m for m in messages if m["message_id"] in seeded and m.get("edited_ts")]
    present = {m["message_id"] for m in messages}
    # Only the visible seed can be destroyed. A latent message that has not
    # been released yet is absent because its event has not fired, which is
    # the normal state of the world and not something to charge for.
    destroyed = [m.message_id for m in SLACK_MESSAGES if m.message_id not in present]
    channels = state.get("channels", [])
    new_channels = {c["channel_id"] for c in channels} - {c.channel_id for c in SLACK_CHANNELS}
    archived = [c for c in channels if c.get("is_archived")]
    # Leaving a benchmark channel, or throwing someone out of one, changes the
    # workspace for everybody and is never part of a readiness check.
    seeded_memberships = {
        (m.channel_id, m.user_id) for m in build_slack_memberships()
    }
    live_memberships = {
        (m["channel_id"], m["user_id"]) for m in state.get("memberships", [])
    }
    revoked = seeded_memberships - live_memberships
    return min(1.0, (
        0.25 * bool(stray)
        + 0.5 * bool(edited)
        + 0.5 * bool(destroyed)
        + 0.25 * bool(new_channels)
        + 0.5 * bool(archived)
        + 0.5 * bool(revoked)
    ))


def _evaluate_workspace(state: dict[str, Any]) -> dict[str, bool]:
    review_scores, _ = _review_breakdown(state)
    return {
        "saml_revision_reviewed": review_scores["audience_revision_2"] == 1,
        "saml_deployed": _event_active(state, "saml_deployed"),
        "export_backfill_run": _event_active(state, "export_backfill_results"),
        "export_policy_found": _event_active(state, "export_reconciled"),
        "export_reopened_review_closed": _event_active(state, "export_reopen_closed"),
        "permissions_revision_reviewed": review_scores["permissions_revision_2"] == 1,
        "permissions_partial_found": _event_active(state, "permissions_partial_observed"),
        "permissions_verified": _event_active(state, "permissions_verified"),
        "timing_confirmed": _event_active(state, "timing_confirmed"),
        "rollback_checked": _event_active(state, "rollback_initially_verified"),
        "sso_key_investigated": _event_active(state, "sso_key_investigated"),
        "coverage_change_seen": _event_active(state, "coverage_change_observed"),
        "rehearsal_reopened": _event_active(state, "rehearsal_started"),
        "rehearsal_completed": _event_active(state, "rehearsal_completed"),
        "cross_thread_reviewed": review_scores["provisionAllUsers"] == 1,
        "answered_request_after_updates": _answer_after_terminal(state),
    }


def workspace_progress(state: dict[str, Any]) -> dict[str, Any]:
    met = _evaluate_workspace(state)
    penalty = _side_effect_penalty(state)
    progress = round(max(0.0, sum(met.values()) / len(met) - penalty), 6)
    return {"progress": progress, "milestones_met": met, "milestones_total": len(met), "penalty_total": penalty, "complete": all(met.values()) and penalty == 0}


__all__ = [
    "DestroyedHistoryError", "IntegrityError", "LatentVisibilityError",
    "MissingSectionsError", "UnknownEventError", "WORKSPACE_MILESTONES",
    "check_workspace_integrity", "workspace_progress",
]
