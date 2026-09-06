#!/usr/bin/env python3
"""Latent-event activation, characterised end to end.

This suite exists to make a failed episode attributable. When an agent finishes
with the workspace unchanged, four very different things may have happened:

  1. it never took an action the rule could consider;
  2. it took a reasonable action the rule rejected;
  3. the rule matched and the event fired, but nothing became observable;
  4. propagation broke somewhere else.

Every test below reports which of those it is observing, and the semantic
matrix deliberately *records* current behaviour rather than asserting the
behaviour we might prefer -- an expected-fail here is a finding, not a bug to
paper over.

Trigger semantics as implemented (read from the seeded rules, not assumed):

  r040_timing_confirmed -> timing_confirmed
      trigger:  reply_keywords
      address:  a reply in CUT001, or a direct message to Daniel (U041) or
                Nina (U046) -- any one of them counts
      actor:    the logged-in agent only
      tools:    reply_to_thread, send_dm_message
      group 1:  one of 9:30 / 930 / nine thirty
      group 2:  one of confirm / question / request / approv / actual / change /
                still / plan / sign / supersede / stand / final
      matching is case-insensitive substring containment, per group, on the body
      of whichever operation carried the ask; every group must match. The
      candidates are stems, so "approv" covers approved and approval. There is
      no semantic understanding: what the gate checks is that the agent noticed
      the 9:30 contradiction and raised it with someone who could settle it.

  Releases on activation: LAT021 (Nina, in CUT001, #acme-migration),
  LAT022 and LAT023 (both in #acme-cutover-bridge, which Ben has not joined),
  and notification NTF-LAT008.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_diagnostics import (
    BEN,
    Action,
    Workspace,
    assert_completely_hidden,
    assert_observable_everywhere,
    render_evaluation,
)
from event_audit import EVENTS, extract_slack_actions
from slack_sim.seed import SCENARIO_RULES

NINA = "U046"
SAM = "U019"
ACME = "C019"
BRIDGE = "C024"
DEBUG = "C023"
DATA = "C021"
SUPPORT = "C022"
LAT021_TEXT = (
    "Confirmed directly with Acme: 9:30 PM was only their VP asking a question, not a change "
    "request. The approved start remains Thursday, August 20 at 9:00 PM PT, with a 90-minute "
    "window and about 15 minutes of customer-visible downtime."
)
CANONICAL = "Nina, can you confirm whether the 9:30 request is real?"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _timing_probe(workspace: Workspace):
    return workspace.probe(
        "LAT021", ACME, thread_parent="CUT001",
        search_terms="Confirmed directly with Acme", notification_id="NTF-LAT008",
    )


# ---------------------------------------------------------------------------
# T0: nothing has happened yet
# ---------------------------------------------------------------------------


def test_at_t0_the_ambiguity_is_present_and_unresolved(root: Path) -> None:
    """CUT001 poses the 9:30 question; CUT015, in its thread, is Ben's stale answer."""
    workspace = Workspace(root)
    root_message = next(
        m for m in workspace.read_channel(ACME).result["messages"] if m["id"] == "CUT001"
    )
    assert "9:30 PM might be safer than 9:00" in root_message["text"]

    replies = workspace.act(
        Action("get_thread_replies",
               {"channel_id": ACME, "thread_ts": str(root_message.get("thread_ts") or root_message["ts"])})
    ).result["messages"]
    stale = next(m for m in replies if m["id"] == "CUT015")
    assert stale["text"] == "Sounds like 9:30 PM is likely for Thursday; updating my notes."
    assert stale["user_id"] == BEN, "the stale assumption must be the agent's own"
    assert workspace.event_status("timing_confirmed") == "pending"


def test_at_t0_lat021_is_invisible_through_every_surface(root: Path) -> None:
    """A pending event must not leak: not in history, threads, search or the inbox."""
    workspace = Workspace(root)
    assert_completely_hidden(_timing_probe(workspace))


def test_at_t0_lat021_is_not_in_the_database_at_all(root: Path) -> None:
    """Latent means unpublished, not hidden by a filter a bug could remove."""
    workspace = Workspace(root)
    assert workspace.stored_messages("LAT021") == []


def test_at_t0_lat021_does_not_inflate_thread_metadata(root: Path) -> None:
    """Reply counts and latest-reply must describe only what the agent can read."""
    workspace = Workspace(root)
    parent = next(m for m in workspace.read_channel(ACME).result["messages"] if m["id"] == "CUT001")
    root_ts = str(parent.get("thread_ts") or parent["ts"])
    replies = workspace.act(
        Action("get_thread_replies", {"channel_id": ACME, "thread_ts": root_ts})
    ).result["messages"]
    visible = [m["id"] for m in replies if m["id"] != "CUT001"]
    assert "LAT021" not in visible
    assert int(parent["reply_count"]) == len(visible), (
        f"reply_count {parent['reply_count']} disagrees with {len(visible)} readable replies"
    )
    assert not parent.get("latest_reply") or any(
        str(m["ts"]) == str(parent["latest_reply"]) for m in replies
    )


def test_at_t0_no_notification_mentions_the_unreleased_reply(root: Path) -> None:
    workspace = Workspace(root)
    inbox = workspace.act(Action("list_notifications", {})).result["notifications"]
    assert not [n for n in inbox if n.get("message_id") == "LAT021"]
    assert not [n for n in inbox if n["notification_id"] == "NTF-LAT008"]


# ---------------------------------------------------------------------------
# The canonical transition
# ---------------------------------------------------------------------------


def test_the_canonical_clarification_activates_the_event_exactly_once(root: Path) -> None:
    workspace = Workspace(root)
    outcome = workspace.reply("CUT001", CANONICAL)

    evaluation = outcome.evaluation("r040_timing_confirmed")
    assert evaluation is not None, "the rule was never even considered"
    assert evaluation.matched, render_evaluation(evaluation)
    assert outcome.activated == ["timing_confirmed"], outcome.activated
    assert workspace.event_status("timing_confirmed") == "activated"

    activation = outcome.trace.activations[0]
    assert activation.status_before == "pending" and activation.status_after == "activated"
    assert set(activation.latent_records_activated) == {"LAT021", "LAT022", "LAT023"}
    assert activation.notifications_activated == ("NTF-LAT008",)
    assert activation.membership_mutations == ()


def test_after_activation_lat021_is_observable_everywhere_it_should_be(root: Path) -> None:
    workspace = Workspace(root)
    workspace.reply("CUT001", CANONICAL)

    visibility = _timing_probe(workspace)
    assert_observable_everywhere(visibility, threaded=True)
    assert visibility.search_visible, visibility.render()
    assert not visibility.history_visible, (
        "channel history returns top-level messages only, for released replies and "
        "the agent's own alike; see test_channel_history_excludes_replies_by_design"
    )
    assert visibility.author == NINA, f"expected Nina, got {visibility.author}"
    assert visibility.body == LAT021_TEXT, "the released text is not the seeded text"


def test_channel_history_excludes_replies_by_design_not_by_a_propagation_bug(root: Path) -> None:
    """Absence from `get_channel_messages` is not evidence that a release failed.

    The tool returns top-level messages only. Ben's own reply is missing from
    history in exactly the same way LAT021 is, so an investigation that reads
    only channel history will mistake correct behaviour for a broken event.
    """
    workspace = Workspace(root)
    mine = workspace.reply("CUT001", CANONICAL).result["id"]
    history = {m["id"] for m in workspace.read_channel(ACME).result["messages"]}

    assert "CUT001" in history, "the thread root is top-level and must be listed"
    assert mine not in history, "the agent's own reply is not in history either"
    assert "LAT021" not in history
    assert _timing_probe(workspace).thread_visible


def test_the_released_reply_lands_in_the_cut001_thread(root: Path) -> None:
    workspace = Workspace(root)
    workspace.reply("CUT001", CANONICAL)

    parent = next(m for m in workspace.read_channel(ACME).result["messages"] if m["id"] == "CUT001")
    replies = workspace.act(
        Action("get_thread_replies",
               {"channel_id": ACME, "thread_ts": str(parent.get("thread_ts") or parent["ts"])})
    ).result["messages"]
    released = next(m for m in replies if m["id"] == "LAT021")
    assert released["thread_id"] == "CUT001"
    assert released["user_id"] == NINA


def test_release_follows_the_action_rather_than_appearing_inside_it(root: Path) -> None:
    """The call that satisfies a rule must not see what it released.

    Scoring depends on this: an agent has to go back and read, which is what
    distinguishes "asked and checked" from "asked".
    """
    workspace = Workspace(root)
    outcome = workspace.reply("CUT001", CANONICAL)
    assert "LAT021" not in json.dumps(outcome.result)
    assert _timing_probe(workspace).thread_visible


def test_repeated_reads_do_not_duplicate_the_release(root: Path) -> None:
    workspace = Workspace(root)
    workspace.reply("CUT001", CANONICAL)
    for _ in range(3):
        workspace.read_channel(ACME)
        workspace.search("Confirmed directly with Acme")
    assert len(workspace.stored_messages("LAT021")) == 1


def test_a_second_qualifying_reply_does_not_release_lat021_twice(root: Path) -> None:
    workspace = Workspace(root)
    first = workspace.reply("CUT001", CANONICAL)
    second = workspace.reply("CUT001", "Confirming again: was 9:30 an actual request?")

    assert first.activated == ["timing_confirmed"]
    assert second.activated == [], "the event fired a second time"
    assert second.evaluation("r040_timing_confirmed") is None, (
        "an activated event should no longer be loaded as a candidate rule"
    )
    assert len(workspace.stored_messages("LAT021")) == 1


def test_the_bridge_releases_are_searchable_but_unreadable_until_ben_joins(root: Path) -> None:
    """LAT022/LAT023 land in a public channel Ben has not joined.

    The two surfaces disagree on purpose, and the disagreement is the intended
    discovery path: search covers public channels whether or not the actor is a
    member, while `get_channel_messages` requires membership. So an agent can
    find the bridge and must then join it to read the thread. "Activated" is
    therefore not the same as "the agent could see it", which matters when
    attributing a failure.
    """
    workspace = Workspace(root)
    workspace.reply("CUT001", CANONICAL)

    before = workspace.probe("LAT022", BRIDGE, search_terms="pinned recovery configuration")
    assert before.search_visible, "search should reach public channels: " + before.render()
    assert not before.history_visible, before.render()
    denied = workspace.read_channel(BRIDGE)
    assert denied.error is not None and "permission_denied" in str(
        getattr(denied.error, "error_code", "")
    ) + str(denied.error), f"reading a non-joined channel should be refused, got {denied.result}"

    workspace.join(BRIDGE)
    after = workspace.probe("LAT022", BRIDGE, search_terms="pinned recovery configuration")
    assert after.history_visible and after.search_visible, after.render()


# ---------------------------------------------------------------------------
# The exact action from the recorded Opus run
# ---------------------------------------------------------------------------


def test_the_recorded_opus_run_contains_no_slack_action_at_all(root: Path) -> None:
    """Replay of the only recorded frontier run, from its captured trajectory.

    Reward 0.03. The trajectory holds four Bash commands and a refusal; it
    contains no Slack tool call, no reply to CUT001, and no mention of 9:30. So
    the timing gate was never exercised: nothing about this run can be
    attributed to the trigger's wording, and the failure sits upstream of the
    workspace entirely. The agent said so itself -- see `final_message`.
    """
    fixture = json.loads((FIXTURES / "opus_4_7_trajectory_actions.json").read_text())
    assert fixture["slack_actions"] == [], "the fixture is stale; re-derive the attribution"
    assert len(fixture["bash_commands"]) == 4
    assert "No MCP or Slack integration is wired into this session" in fixture["final_message"]

    # Anything the agent did do is replayed for completeness. None of it is a
    # Slack action, so none of it can reach the evaluator.
    workspace = Workspace(root)
    for command in fixture["bash_commands"]:
        assert "slack" not in command.split(), command
    assert workspace.event_status("timing_confirmed") == "pending"


def test_the_extractor_would_have_found_a_slack_action_if_one_existed(root: Path) -> None:
    """Proves the empty result above is a fact about the run, not a broken parser."""
    synthetic = {
        "steps": [
            {"tool_calls": [{"name": "mcp__slack__reply_to_thread",
                             "arguments": {"thread_parent_id": "CUT001", "body": CANONICAL}}]},
            {"tool_calls": [{"name": "Bash", "arguments": {
                "command": "slack reply_to_thread --thread-parent-id CUT001 --body 'Was 9:30 an actual request?'"}}]},
        ]
    }
    path = root / "synthetic-trajectory.json"
    path.write_text(json.dumps(synthetic))

    actions = extract_slack_actions(path)
    assert [a.tool for a in actions] == ["reply_to_thread", "reply_to_thread"]
    assert actions[0].payload["thread_parent_id"] == "CUT001"
    assert actions[1].payload["body"] == "Was 9:30 an actual request?"

    workspace = Workspace(root)
    assert workspace.act(actions[0]).activated == ["timing_confirmed"]


# ---------------------------------------------------------------------------
# Semantic matrix: how brittle is the gate?
# ---------------------------------------------------------------------------

#: Phrasings a competent engineer might use to resolve the same ambiguity.
#: The two that are still rejected are rejected on purpose: neither references
#: the 9:30 contradiction, so neither shows the agent read CUT001. Everything
#: that does reference it is now accepted, including the three that used to
#: fail on a missing synonym or a word stem.
CLARIFICATION_PHRASINGS: tuple[tuple[str, bool], ...] = (
    ("Nina, can you confirm whether the 9:30 request is real?", True),
    ("Is 9:30 still the plan?", True),
    ("Did the customer actually move the start to 9:30?", True),
    ("Was 9:30 a change or just a question?", True),
    ("Do we still have approval for 9:00, or has the start changed to 9:30?", True),
    ("Can someone reconfirm where the migration time stands? I saw the 9:30 note.", True),
    ("Did 9:30 supersede the approved start?", True),
    ("Has the 9:30 ask been signed off?", True),
    # Rejected: asks about the schedule without engaging the contradiction.
    ("What time are we actually starting Thursday?", False),
    ("Nina, please check with Acme whether the start time moved.", False),
)


def test_the_semantic_matrix_matches_recorded_behaviour(root: Path) -> None:
    """Characterisation, printed as a table, of ten plausible clarifications.

    A row that stops matching this table is a deliberate change to how much the
    gate demands, and should be justified rather than re-recorded.
    """
    print("\n    phrasing                                                          fires  why not")
    surprises = []
    for index, (phrasing, expected) in enumerate(CLARIFICATION_PHRASINGS):
        workspace = Workspace(root / f"phrase{index}")
        outcome = workspace.reply("CUT001", phrasing)
        evaluation = outcome.evaluation("r040_timing_confirmed")
        fired = bool(outcome.activated)
        failed = ", ".join(evaluation.failed_predicates) if evaluation else "not considered"
        print(f"    {phrasing[:62]:64} {str(fired):5}  {'' if fired else failed}")
        if fired != expected:
            surprises.append((phrasing, expected, fired))
        assert fired == (workspace.event_status("timing_confirmed") == "activated")
        if fired:
            assert _timing_probe(workspace).thread_visible, "fired but nothing became visible"

    assert not surprises, f"recorded behaviour changed: {surprises}"


def test_a_rejected_phrasing_fails_only_on_the_keyword_group(root: Path) -> None:
    """What the gate still insists on, and the precise diagnosis when it refuses.

    "What time are we actually starting Thursday?" is addressed to the right
    thread by the right actor and asks about the schedule -- but it never
    references the 9:30 contradiction, so it is the question of someone who has
    not read CUT001. That is the thing the gate exists to check, and it is the
    only thing it fails on here.

    Its former companion, "Is 9:30 still the plan?", is now accepted: see
    tests/test_trigger_fairness.py.
    """
    workspace = Workspace(root)
    outcome = workspace.reply("CUT001", "What time are we actually starting Thursday?")
    evaluation = outcome.evaluation("r040_timing_confirmed")
    assert evaluation is not None

    passed = {p.name for p in evaluation.predicates if p.passed}
    assert {"action_type", "actor_is_the_agent", "address", "keyword_group_2"} <= passed
    assert evaluation.failed_predicates == ("keyword_group_1",), render_evaluation(evaluation)

    group = next(p for p in evaluation.predicates if p.name == "keyword_group_1")
    assert group.detail["matched"] == []
    assert "9:30" in group.detail["candidates"]
    assert_completely_hidden(_timing_probe(workspace))


# ---------------------------------------------------------------------------
# The Nina DM path
# ---------------------------------------------------------------------------


def test_asking_nina_directly_now_counts_and_the_trace_says_where_it_landed(root: Path) -> None:
    """The prompt says to ask the relevant person; Nina is the relevant person.

    This used to be rejected. The rule listened to one thread, so a DM carrying
    the identical question -- right actor, right words, right person -- failed
    on addressing while its keyword predicates reported `matched=[]`, which made
    an addressing failure read like a wording failure. Both are fixed: the rule
    now declares the people it will accept, and the trace names the address that
    matched.
    """
    workspace = Workspace(root)
    outcome = workspace.dm(
        NINA, "Was the 9:30 PM ask an actual change request, or just a question? Please confirm."
    )

    evaluation = outcome.evaluation("r040_timing_confirmed")
    assert evaluation is not None and evaluation.matched, render_evaluation(evaluation)
    assert evaluation.action == "send_dm_message"

    address = next(p for p in evaluation.predicates if p.name == "address")
    assert address.detail["matched"] == [f"dm:{NINA}"]
    assert f"thread:CUT001" in address.detail["accepted"]
    assert workspace.event_status("timing_confirmed") == "activated"


def test_a_dm_to_the_wrong_person_does_not_settle_the_timing(root: Path) -> None:
    """Widening addressing is not the same as removing it.

    Sam owns rollback, not the customer relationship. Asking him about the
    start time demonstrates nothing about who can settle it, so the rule keeps
    refusing -- and the trace says which addresses it would have accepted.
    """
    workspace = Workspace(root)
    outcome = workspace.dm(SAM, "Was the 9:30 PM ask an actual change request?")

    evaluation = outcome.evaluation("r040_timing_confirmed")
    assert evaluation is not None and not evaluation.matched
    address = next(p for p in evaluation.predicates if p.name == "address")
    assert address.detail["matched"] == []
    assert address.detail["actual_recipient"] == SAM
    assert workspace.event_status("timing_confirmed") == "pending"
    assert_completely_hidden(_timing_probe(workspace))


def test_the_answer_to_a_dm_ask_lands_in_the_thread_not_the_dm(root: Path) -> None:
    """Nina answers where the question was raised publicly, and the inbox says so.

    Worth pinning down, because an agent that asks by DM and then watches only
    the DM will conclude nobody replied. The notification is what makes the
    answer findable, which is why it is asserted here rather than assumed.
    """
    workspace = Workspace(root)
    outcome = workspace.dm(NINA, "Was the 9:30 PM ask an actual change request?")
    chat_id = outcome.result["message"]["channel_id"]

    conversation = workspace.act(Action("get_channel_messages", {"channel_id": chat_id})).result
    assert {m["user_id"] for m in conversation["messages"]} == {BEN}, "Nina replied in the DM"

    visibility = _timing_probe(workspace)
    assert visibility.thread_visible, "the answer must exist somewhere the agent can reach"
    assert visibility.notification_visible, "and the inbox must point at it"


# ---------------------------------------------------------------------------
# Negative cases: the gate must not fire on mere mentions
# ---------------------------------------------------------------------------

NON_TRIGGERS: tuple[tuple[str, str, str], ...] = (
    ("reply in CUT001 agreeing", "CUT001", "9:30 works for me."),
    ("reply in CUT001 about a chart", "CUT001", "The 9:30 graph looked better."),
    ("unrelated approval question", "MSG129", "Was that request approved?"),
)


def test_unrelated_activity_never_confirms_the_timing(root: Path) -> None:
    for index, (label, thread, body) in enumerate(NON_TRIGGERS):
        workspace = Workspace(root / f"neg{index}")
        outcome = workspace.reply(thread, body)
        assert outcome.activated == [], f"{label} activated {outcome.activated}"
        assert workspace.event_status("timing_confirmed") == "pending"
        assert_completely_hidden(_timing_probe(workspace))


def test_a_maintenance_notice_in_another_channel_is_inert(root: Path) -> None:
    workspace = Workspace(root)
    outcome = workspace.post("C014", "Our database maintenance is at 9:30.")
    assert outcome.activated == []
    assert workspace.event_status("timing_confirmed") == "pending"


def test_an_unrelated_dm_to_nina_is_inert(root: Path) -> None:
    workspace = Workspace(root)
    outcome = workspace.dm(NINA, "Can you confirm the export count?")
    assert outcome.activated == []
    assert workspace.event_status("timing_confirmed") == "pending"


def test_a_qualifying_body_in_the_wrong_thread_is_inert(root: Path) -> None:
    """Isolates the thread predicate from the wording predicates."""
    workspace = Workspace(root)
    outcome = workspace.reply("MSG129", CANONICAL)
    evaluation = outcome.evaluation("r040_timing_confirmed")
    assert evaluation is not None
    failed = set(evaluation.failed_predicates)
    assert failed == {"address"}, render_evaluation(evaluation)


# ---------------------------------------------------------------------------
# The same skeleton, applied to the other consequential transitions
# ---------------------------------------------------------------------------


def _drive(workspace: Workspace, event_id: str) -> None:
    spec = EVENTS[event_id]
    for required in spec.get("requires", ()):
        _drive(workspace, required)
    if event_id == "rollback_initially_verified":
        workspace.join(BRIDGE)
    workspace.reply(str(spec["canonical_thread"]), str(spec["canonical_body"]))


def test_each_consequential_transition_hides_then_reveals_its_payload(root: Path) -> None:
    """T0 invisible -> canonical action -> activated -> observable -> no duplication.

    One skeleton over every gate that moves the readiness picture, so a
    regression in any of them is attributable to the same four questions as the
    timing gate.
    """
    for index, event_id in enumerate(
        ("saml_revision_2", "saml_deployed", "export_backfill_results",
         "permissions_revision_2", "permissions_checker_rerun", "rollback_initially_verified")
    ):
        spec = EVENTS[event_id]
        workspace = Workspace(root / f"event{index}")
        message_id = str(spec["message_id"])
        channel = str(spec["channel"])
        thread_parent = spec["thread_parent"]
        search_terms = str(spec["search_terms"])

        for required in spec.get("requires", ()):
            _drive(workspace, required)
        if event_id == "rollback_initially_verified":
            workspace.join(BRIDGE)

        assert workspace.event_status(event_id) == "pending", event_id
        assert_completely_hidden(
            workspace.probe(message_id, channel, thread_parent, search_terms)
        )

        outcome = workspace.reply(str(spec["canonical_thread"]), str(spec["canonical_body"]))
        assert event_id in outcome.activated, (
            f"{event_id} did not activate; considered rules: "
            f"{[e.rule_id for e in outcome.trace.evaluations]}"
        )
        assert workspace.event_status(event_id) == "activated"

        visibility = workspace.probe(
            message_id, channel, thread_parent, search_terms,
            notification_id=spec["notification_id"],
        )
        assert visibility.search_visible, f"{event_id}: {visibility.render()}"
        assert_observable_everywhere(
            visibility,
            threaded=bool(thread_parent),
            expect_notification=bool(spec["notification_id"]),
        )
        assert len(workspace.stored_messages(message_id)) == 1
        print(f"    {event_id:30} -> {message_id} observable")


def test_observation_gated_events_need_the_agent_to_have_read_the_evidence(root: Path) -> None:
    """`observed` rules fire on what a result actually contained.

    The distinction matters for attribution: an agent can hold the right belief
    and still not advance the world, because these gates measure reading.
    """
    workspace = Workspace(root)
    assert workspace.event_status("sso_context_observed") == "pending"

    workspace.search("nothing matches this string at all")
    assert workspace.event_status("sso_context_observed") == "pending"

    outcome = workspace.search("IDP-ACME-014")
    evaluation = outcome.evaluation("r100_sso_context")
    assert evaluation is not None
    predicate = next(p for p in evaluation.predicates if p.name == "observed_required_evidence")
    assert predicate.passed, predicate.detail
    assert workspace.event_status("sso_context_observed") == "activated"


def test_the_coverage_membership_change_becomes_visible_after_it_is_read(root: Path) -> None:
    """Staffing: reading LAT023 puts the identity backup into the bridge channel."""
    workspace = Workspace(root)
    workspace.reply("CUT001", CANONICAL)
    workspace.join(BRIDGE)

    members = workspace.act(
        Action("list_channel_members", {"channel_id": BRIDGE})
    ).result["members"]
    assert "U051" not in {m["user_id"] for m in members}
    assert workspace.event_status("coverage_change_observed") == "pending"

    parent = next(m for m in workspace.read_channel(BRIDGE).result["messages"] if m["id"] == "CUT002")
    outcome = workspace.act(
        Action("get_thread_replies",
               {"channel_id": BRIDGE, "thread_ts": str(parent.get("thread_ts") or parent["ts"])})
    )
    assert "coverage_change_observed" in outcome.activated
    assert outcome.trace.activations[0].membership_mutations == ("MBR-LAT001",)

    members = workspace.act(
        Action("list_channel_members", {"channel_id": BRIDGE})
    ).result["members"]
    assert "U051" in {m["user_id"] for m in members}


def test_the_export_review_reopens_with_a_newer_revision_to_review(root: Path) -> None:
    """A closed review must visibly reopen, or the agent cannot know to return."""
    workspace = Workspace(root)
    workspace.reply("MSG200", "Looks good to me")
    assert workspace.event_status("export_backfill_results") == "activated"

    workspace.search("difference 26")
    workspace.search("26 synthetic")
    outcome = workspace.search("Reconciliation complete")
    assert "export_reconciled" in outcome.activated or (
        workspace.event_status("export_reconciled") == "activated"
    )

    reopened = workspace.probe("LAT011", DEBUG, thread_parent="MSG200",
                               search_terms="reconciliation assertion")
    assert reopened.thread_visible, reopened.render()
    assert workspace.event_status("export_reopen_closed") == "pending", (
        "the reopened review must still be open until it is approved again"
    )

    workspace.reply("MSG200", "Looks good to me")
    assert workspace.event_status("export_reopen_closed") == "activated"
    assert len(workspace.stored_messages("LAT017")) == 1


# ---------------------------------------------------------------------------
# Invariants that should hold for any latent event, not just these
# ---------------------------------------------------------------------------


def test_no_pending_event_leaks_any_of_its_payload(root: Path) -> None:
    """Sweep every latent message of every still-pending event."""
    from slack_sim.seed import SLACK_LATENT_MESSAGES

    workspace = Workspace(root)
    statuses = workspace.event_statuses()
    hidden = [m for m in SLACK_LATENT_MESSAGES if statuses.get(m.event_id) == "pending"]
    assert hidden, "no pending latent messages: the fixture changed shape"
    for message in hidden:
        assert workspace.stored_messages(message.message_id) == [], message.message_id
        assert not workspace.probe(message.message_id, message.conversation_id).anywhere


def test_no_event_can_fire_before_the_messages_it_replies_to_exist(root: Path) -> None:
    """Ordering must be declared, never implied by a thread id.

    A latent reply whose parent is itself latent can only be published after
    the parent's event. While a rule is thread-only that holds by accident --
    nobody can reply in a thread that does not exist yet -- so the ordering
    goes unstated. Widen that rule to accept a channel or a DM and the accident
    is gone: the event fires early, publishing a reply to a missing parent
    raises inside the tool call, and the whole transaction rolls back, taking a
    legitimate activation with it. This is exactly how `r060_rehearsal_closed`
    broke when the rollback ask was widened.
    """
    from slack_sim.seed import SLACK_LATENT_MESSAGES, SLACK_MESSAGES

    owner = {m.message_id: m.event_id for m in SLACK_LATENT_MESSAGES}
    rules_for: dict[str, list] = {}
    for rule in SCENARIO_RULES:
        rules_for.setdefault(rule.event_id, []).append(rule)

    def guaranteed_before(event_id: str, seen: frozenset[str] = frozenset()) -> set[str]:
        """Events that must already be activated before `event_id` can fire."""
        if event_id in seen:
            return set()
        seen = seen | {event_id}
        per_rule = []
        for rule in rules_for.get(event_id, []):
            reached: set[str] = set()
            for required in rule.requires_activated:
                reached.add(required)
                reached |= guaranteed_before(required, seen)
            per_rule.append(reached)
        if not per_rule:
            return set()
        return set.intersection(*per_rule)

    seeded = {m.message_id for m in SLACK_MESSAGES}
    checked = 0
    for message in SLACK_LATENT_MESSAGES:
        for parent in (message.thread_parent_id, message.reply_to_id):
            if not parent or parent in seeded:
                continue
            assert parent in owner, f"{message.message_id} replies to unknown {parent}"
            producer = owner[parent]
            if producer == message.event_id:
                continue  # published together, in ordinal order
            checked += 1
            assert producer in guaranteed_before(message.event_id), (
                f"{message.event_id} publishes {message.message_id} as a reply to "
                f"{parent}, which only exists once {producer} has fired -- but no "
                f"rule for {message.event_id} requires it. Add "
                f"requires_activated=({producer!r},)."
            )
    assert checked, "no cross-event latent replies found: the fixture changed shape"


def test_every_rule_names_an_event_that_exists_and_can_still_be_reached(root: Path) -> None:
    workspace = Workspace(root)
    statuses = workspace.event_statuses()
    for rule in SCENARIO_RULES:
        assert rule.event_id in statuses, f"{rule.rule_id} gates an unknown event"
        for required in (*rule.requires_activated, *rule.requires_pending):
            assert required in statuses, f"{rule.rule_id} depends on unknown event {required}"


def main() -> None:
    tests = sorted(
        (value for name, value in globals().items()
         if name.startswith("test_") and callable(value)),
        key=lambda fn: fn.__name__,
    )
    with tempfile.TemporaryDirectory() as directory:
        for index, test in enumerate(tests):
            root = Path(directory) / str(index)
            root.mkdir()
            test(root)
            print(f"  {test.__name__}: ok")
    print(f"event activation: ok ({len(tests)} tests)")


if __name__ == "__main__":
    main()
