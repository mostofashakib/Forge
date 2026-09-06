#!/usr/bin/env python3
"""Dynamic transition and terminal grading tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

from dynamic_scenario import call, drive_terminal, seeded
import test_migration_readiness as entry
from slack_sim.clock import VIRTUAL_CLOCK, moment
from slack_sim.migration_truth import REQUIRED_EVENTS
from slack_sim.service import export_state


def events(db: Path) -> dict[str, str]:
    return {event["event_id"]: event["status"] for event in export_state(db)["scenario_events"]}


def test_wrong_review_does_not_release_future_state(root: Path) -> None:
    db = seeded(root)
    call(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "Looks good to me"})
    state = export_state(db)
    assert events(db)["saml_revision_2"] == "pending"
    assert not any(message["message_id"].startswith("LAT") for message in state["messages"])


def test_saml_requires_critique_then_second_review(root: Path) -> None:
    db = seeded(root)
    critique = "includes is substring matching, but exact equality is required; a partial audience passes incorrectly"
    call(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": critique})
    state = export_state(db)
    assert events(db)["saml_revision_2"] == "activated"
    assert "LAT001" in {m["message_id"] for m in state["messages"]}
    assert events(db)["saml_deployed"] == "pending"
    call(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "Looks good to me"})
    ids = {m["message_id"] for m in export_state(db)["messages"]}
    assert events(db)["saml_deployed"] == "activated"
    assert {"LAT002", "LAT003", "LAT004", "LAT005"} <= ids


def test_audience_matches_public_visibility_sequence(root: Path) -> None:
    """The audience fix and its operational outcome appear only when earned."""
    db = seeded(root)  # RESET: create a fresh episode from the T0 fixture.

    discovery = call(db, "search_messages", {"query": "audienceMatches"})
    root_message = next(
        message for message in discovery["matches"] if message["id"] == "MSG194"
    )
    thread_payload = {
        "channel_id": "C023",
        "thread_ts": root_message["thread_ts"],
    }

    initial_thread = call(db, "get_thread_replies", thread_payload)
    initial_thread_ids = {message["id"] for message in initial_thread["messages"]}
    initial_identity = call(db, "get_channel_messages", {"channel_id": "C020"})
    initial_identity_ids = {message["id"] for message in initial_identity["messages"]}

    # T0 exposes revision 1 and none of the latent revision/deployment outcome.
    assert {"MSG194", "MSG197"} <= initial_thread_ids
    assert {"LAT001", "LAT002"}.isdisjoint(initial_thread_ids)
    assert {"LAT003", "LAT004", "LAT005"}.isdisjoint(initial_identity_ids)

    critique = call(
        db,
        "reply_to_thread",
        {
            "thread_parent_id": "MSG194",
            "body": (
                "The implementation uses includes/substring matching, but the SAML "
                "audience requirement is exact equality. Use === after canonicalization."
            ),
        },
    )
    after_critique = call(db, "get_thread_replies", thread_payload)
    after_critique_ids = {message["id"] for message in after_critique["messages"]}
    identity_after_critique = call(
        db, "get_channel_messages", {"channel_id": "C020"}
    )

    # Revision 1 remains history, revision 2 is now readable, but CI and the
    # operational outcome still have not been published.
    assert {"MSG197", critique["id"], "LAT001"} <= after_critique_ids
    assert "LAT002" not in after_critique_ids
    assert {"LAT003", "LAT004", "LAT005"}.isdisjoint(
        {message["id"] for message in identity_after_critique["messages"]}
    )

    approval = call(
        db,
        "reply_to_thread",
        {"thread_parent_id": "MSG194", "body": "Looks good to me"},
    )
    after_approval = call(db, "get_thread_replies", thread_payload)
    after_approval_ids = {message["id"] for message in after_approval["messages"]}

    # The final review read includes revision 2, Ben's approval, and the CI
    # result. Deployment and smoke-test evidence live in identity-eng instead.
    assert {"LAT001", approval["id"], "LAT002"} <= after_approval_ids
    assert {"LAT003", "LAT004", "LAT005"}.isdisjoint(after_approval_ids)

    identity_after_approval = call(
        db, "get_channel_messages", {"channel_id": "C020"}
    )
    operational_ids = {
        message["id"] for message in identity_after_approval["messages"]
    }
    assert {"LAT003", "LAT004", "LAT005"} <= operational_ids
    signature_failure = next(
        message
        for message in identity_after_approval["messages"]
        if message["id"] == "LAT004"
    )
    assert "EU-2 FAIL" in signature_failure["text"]
    assert "signature verification fails" in signature_failure["text"]
    assert "acme-eu-2025" in signature_failure["text"]


def test_export_policy_transition_requires_actual_observation(root: Path) -> None:
    db = seeded(root)
    call(db, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})
    assert events(db)["export_backfill_results"] == "activated"
    call(db, "search_messages", {"query": "operations note", "limit": 1})
    assert events(db)["export_reconciled"] == "pending", "internally scanned rows must not count as observed"
    call(db, "search_messages", {"query": "26 synthetic", "limit": 50})
    assert events(db)["export_reconciled"] == "activated"
    assert events(db)["export_reopen_closed"] == "pending"
    call(db, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})
    assert events(db)["export_reopen_closed"] == "activated"


def test_permissions_progresses_through_invalidated_green_to_verification(root: Path) -> None:
    db = seeded(root)
    call(db, "search_messages", {"query": "ACME-ACCESS-04"})
    critique = "some is any-match and accepts one entitlement; the policy requires every/all required group"
    call(db, "reply_to_thread", {"thread_parent_id": "MSG214", "body": critique})
    assert events(db)["permissions_revision_2"] == "activated"
    assert events(db)["permissions_verified"] == "pending"
    call(db, "reply_to_thread", {"thread_parent_id": "MSG214", "body": "Looks good to me"})
    assert events(db)["permissions_checker_rerun"] == "activated"
    assert events(db)["permissions_partial_verification"] == "pending"

    checker_page = call(db, "get_channel_messages", {"channel_id": "C022"})
    assert {"LAT013", "LAT014"} <= {message["id"] for message in checker_page["messages"]}
    assert events(db)["permissions_partial_verification"] == "activated"
    assert events(db)["permissions_verified"] == "pending"

    partial_page = call(db, "get_channel_messages", {"channel_id": "C022"})
    assert "LAT015" in {message["id"] for message in partial_page["messages"]}
    assert "LAT020" not in {message["id"] for message in partial_page["messages"]}
    assert events(db)["permissions_verified"] == "activated"

    verified_page = call(db, "get_channel_messages", {"channel_id": "C022"})
    final_ids = {message["id"] for message in verified_page["messages"]}
    assert "LAT020" in final_ids
    acme_page = call(db, "get_channel_messages", {"channel_id": "C019"})
    assert "LAT016" in {message["id"] for message in acme_page["messages"]}
    bodies = "\n".join(message["text"] for message in verified_page["messages"])
    assert "EU-Legal FAIL" in "\n".join(message["text"] for message in checker_page["messages"])
    assert "restricted export area" in bodies and "export action successfully" in bodies


def test_timing_rollback_and_rehearsal_require_reobservation(root: Path) -> None:
    db = seeded(root)
    state = drive_terminal(db, answer=False)
    assert events(db)["timing_confirmed"] == "activated"
    assert events(db)["coverage_change_observed"] == "activated"
    assert events(db)["rollback_initially_verified"] == "activated"
    assert events(db)["rehearsal_started"] == "activated"
    assert events(db)["rehearsal_completed"] == "activated"

    by_id = {message["message_id"]: message for message in state["messages"]}
    assert "9:30 PM was only" in by_id["LAT021"]["body"]
    assert "stale" in by_id["LAT031"]["body"]
    assert "PASS" in by_id["LAT033"]["body"]
    assert "remaining EU-2 key" in by_id["LAT035"]["body"]
    assert by_id["LAT027"]["ts"] == VIRTUAL_CLOCK.at(moment(16, 19, 30))


def test_transition_activation_is_idempotent(root: Path) -> None:
    db = seeded(root)
    critique = "includes performs a substring/partial comparison; exact equality is required"
    call(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": critique})
    call(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": critique})
    state = export_state(db)
    assert sum(message["message_id"] == "LAT001" for message in state["messages"]) == 1


def test_perfect_terminal_episode_scores_one(root: Path) -> None:
    db = seeded(root)
    state = drive_terminal(db)
    rewards, details = entry.evaluate(state)
    assert {event_id for event_id, status in events(db).items()
            if status == "activated"} == REQUIRED_EVENTS
    assert rewards["reward"] == 1.0, details
    assert rewards["success"] == 1.0, details


def test_t0_or_premature_answer_cannot_score_full(root: Path) -> None:
    db = seeded(root)
    call(db, "reply_to_thread", {"thread_parent_id": "MSG145",
                                 "body": "BLOCKED at 9 PM; all initial issues remain blocked."})
    rewards, details = entry.evaluate(export_state(db))
    assert rewards["success"] == 0.0
    assert rewards["reward"] < 1.0
    assert "answered_after:" in details["summary"], details["summary"]


def test_a_terminal_episode_that_rests_on_t0_evidence_loses_credit(root: Path) -> None:
    """The same property the old artifact test asserted, now about the report
    the prompt actually asks for: a reply that reaches the terminal state and
    then describes the world as it was at t0."""
    db = seeded(root)
    drive_terminal(db, answer=False)
    call(db, "reply_to_thread", {
        "thread_parent_id": "MSG145",
        "body": "BLOCKED. The audience validation bug is still the SSO blocker (MSG146). "
                "Export incomplete per MSG137. Permissions done per MSG143.",
    })
    rewards, details = entry.evaluate(export_state(db))
    assert rewards["success"] == 0.0
    assert 0.0 < rewards["reward"] < 1.0, details["summary"]
    charged = [line for line in details["summary"].splitlines() if "superseded" in line]
    assert charged, details["summary"]


# ---------------------------------------------------------------------------
# What an invalidated episode is allowed to say about itself
# ---------------------------------------------------------------------------
#
# A failed integrity check zeroes the run and sets valid=0, so the exception is
# the entire record of why an episode was thrown away. One class and a sentence
# made every cause look alike: a malformed export (the harness is broken), a
# destroyed message (the store was damaged), a visibility mismatch (the scenario
# engine disagrees with its own ledger) and an unknown event (the ledger was
# rewritten) each call for a different person to look at something different.


def test_a_malformed_export_names_the_sections_it_lacks(root: Path) -> None:
    from slack_sim.migration_reward import (
        IntegrityError, MissingSectionsError, check_workspace_integrity,
    )

    try:
        check_workspace_integrity({"users": [], "channels": []})
    except MissingSectionsError as error:
        assert isinstance(error, IntegrityError)
        assert error.kind == "missing_sections"
        assert "messages" in error.sections and "reactions" in error.sections
        assert error.sections == sorted(error.sections), "evidence must be ordered"
        return
    raise AssertionError("an export missing half its sections passed the check")


def test_destroyed_history_names_the_messages_that_vanished(root: Path) -> None:
    from slack_sim.migration_reward import (
        DestroyedHistoryError, check_workspace_integrity,
    )

    db = seeded(root)
    state = export_state(db)
    state["messages"] = [m for m in state["messages"] if m["message_id"] != "MSG135"]
    try:
        check_workspace_integrity(state)
    except DestroyedHistoryError as error:
        assert error.kind == "destroyed_history"
        assert "MSG135" in error.message_ids
        return
    raise AssertionError("destroyed seeded history passed the check")


def test_a_rewritten_ledger_names_the_event_it_invented(root: Path) -> None:
    from slack_sim.migration_reward import UnknownEventError, check_workspace_integrity

    db = seeded(root)
    state = export_state(db)
    state["scenario_events"] = list(state["scenario_events"]) + [
        {"event_id": "invented_event", "status": "activated"},
    ]
    try:
        check_workspace_integrity(state)
    except UnknownEventError as error:
        assert error.kind == "unknown_event"
        assert error.event_ids == ["invented_event"], (
            "the report must name the invented event, not merely that one exists"
        )
        return
    raise AssertionError("an event the task never defines passed the check")


def test_a_latent_message_released_without_its_event_is_reported_as_such(root: Path) -> None:
    from slack_sim.migration_reward import LatentVisibilityError, check_workspace_integrity

    db = seeded(root)
    state = export_state(db)
    # LAT001 is released by the saml_revision_2 event, which has not fired.
    state["messages"] = list(state["messages"]) + [
        {"message_id": "LAT001", "author_id": "U002", "ts": "1.0", "body": "smuggled"},
    ]
    try:
        check_workspace_integrity(state)
    except LatentVisibilityError as error:
        assert error.kind == "latent_visibility"
        assert error.unexpected == ["LAT001"]
        assert error.missing == []
        return
    raise AssertionError("a latent message released without its event passed the check")


def test_every_integrity_failure_still_arrives_as_an_integrity_error(root: Path) -> None:
    """The Harbor verifier imports exactly one name for this and catches it by
    that name. A subclass that escaped the base would reach the reward file as
    an unhandled exception instead of `valid=0`."""
    from slack_sim import migration_reward as module

    subclasses = [
        value for name, value in vars(module).items()
        if isinstance(value, type) and issubclass(value, module.IntegrityError)
        and value is not module.IntegrityError
    ]
    assert len(subclasses) >= 4, f"only found {[c.__name__ for c in subclasses]}"
    for cls in subclasses:
        assert cls.kind, f"{cls.__name__} has no kind"
        assert cls.__name__ in module.__all__, f"{cls.__name__} is not exported"


def test_an_integrity_failure_reports_itself_as_data(root: Path) -> None:
    """`str(error)` is a sentence for a human. `as_dict()` is for the reward
    file, where a reader is trying to tell one cause from another."""
    from slack_sim.migration_reward import MissingSectionsError

    error = MissingSectionsError(["messages", "reactions"])
    assert error.as_dict() == {
        "kind": "missing_sections",
        "message": str(error),
        "sections": ["messages", "reactions"],
    }
    assert "messages" in str(error), "the sentence must still name the evidence"


def test_a_discarded_run_records_why_as_data(root: Path) -> None:
    """`valid=0` with a sentence tells a reader the run was thrown away but not
    what to go and look at. The kind and its evidence belong in the details."""
    db = seeded(root)
    state = export_state(db)
    state["messages"] = [m for m in state["messages"] if m["message_id"] != "MSG135"]
    rewards, details = entry.evaluate(state)

    assert rewards["valid"] == 0.0
    assert details["integrity"]["kind"] == "destroyed_history"
    assert "MSG135" in details["integrity"]["message_ids"]
    # The old key keeps its old shape; anything already reading it still works.
    assert details["integrity_error"] == details["integrity"]["message"]


def test_a_grader_that_breaks_is_not_reported_as_a_damaged_workspace(root: Path) -> None:
    """The catch-all below the typed branch means the check itself crashed.
    Labelling that `destroyed_history` would send someone hunting a workspace
    that was never touched."""
    rewards, details = entry.evaluate({
        "users": [], "channels": [], "reactions": [], "chats": [],
        "scenario_events": [],
        # Every section is present, so the check gets past its own guard and
        # then trips over a row that has no id.
        "messages": [{"body": "a message-shaped thing with no message_id"}],
    })
    assert rewards["valid"] == 0.0
    assert details["integrity"]["kind"] == "check_failed"


def main() -> None:
    tests = sorted((value for name, value in globals().items() if name.startswith("test_") and callable(value)), key=lambda fn: fn.__name__)
    with tempfile.TemporaryDirectory() as directory:
        for index, test in enumerate(tests):
            test(Path(directory) / str(index))
            print(f"  {test.__name__}: ok")
    print("migration reward: ok")


if __name__ == "__main__":
    main()
