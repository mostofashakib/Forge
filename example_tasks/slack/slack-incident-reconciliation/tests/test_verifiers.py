#!/usr/bin/env python3
"""The verifier stack, checked the way it will be used.

Two kinds of test live here. Most build tiny synthetic episodes with no Acme
content at all, because the framework must not know which task it is grading.
The last few drive the real oracle solution end to end, because a contract that
disagrees with a known-correct episode is wrong about the task no matter how
clean its unit tests are -- and the first run of this contract did disagree,
twice.
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verifiers import (
    EventVerifier,
    ExactStateVerifier,
    LayeredVerifier,
    NegativeVerifier,
    PolicyVerifier,
    REWARD_PRESETS,
    RewardHackingAuditor,
    TemporalVerifier,
    TieredRewardEngine,
    VerifierComposer,
    load_experiment,
)
from verifiers.checks import (
    ActionPenalty,
    Contract,
    Forbidden,
    Ordering,
    Policy,
    StateExpectation,
    ToolExpectation,
    TrajectoryVerifier,
)
import verifiers as verifiers_package
from verifiers.episode import Episode

#: Locate the package as imported, not by a path relative to this file: in the
#: container these tests live at /tests and the package at /opt/grading.
PACKAGE = Path(verifiers_package.__file__).resolve().parent
ACTOR = "U002"


# ---------------------------------------------------------------------------
# Synthetic episodes: nothing here knows about Acme
# ---------------------------------------------------------------------------


def _episode(
    events: dict[str, str] | None = None,
    actions: list[dict] | None = None,
    messages: list[dict] | None = None,
    with_log: bool = True,
) -> Episode:
    state: dict = {
        "scenario_events": [
            {"event_id": name, "status": status, "activated_ts": ts}
            for name, (status, ts) in (events or {}).items()
        ],
        "messages": messages or [],
        "channels": [],
        "memberships": [],
    }
    if with_log:
        state["action_log"] = [
            {"seq": index + 1, "ts": str(1000 + index), "actor_id": ACTOR, **action}
            for index, action in enumerate(actions or [])
        ]
    return Episode.from_state(state)


def _events(**pairs: tuple[str, str]) -> dict[str, tuple[str, str]]:
    return dict(pairs)


def test_exact_state_verifier_reports_the_value_it_found(root: Path) -> None:
    episode = _episode(messages=[{"message_id": "M1", "author_id": ACTOR}])
    verifier = ExactStateVerifier(
        expectations=(
            StateExpectation("message_count", lambda e: len(e.messages), 1),
            StateExpectation("wrong", lambda e: len(e.messages), 7),
        )
    )
    result = verifier.verify(episode)
    assert result.score == 0.5 and not result.passed
    assert result.failed_checks == ("wrong",)
    assert dict(result.checks[1].detail)["actual"] == 1


def test_a_broken_selector_fails_its_check_instead_of_the_run(root: Path) -> None:
    verifier = ExactStateVerifier(
        expectations=(StateExpectation("boom", lambda e: e.missing_attribute, 1),)
    )
    result = verifier.verify(_episode())
    assert not result.passed
    assert "AttributeError" in result.checks[0].detail["error"]


def test_policy_expressions_can_use_contract_names_inside_comprehensions(root: Path) -> None:
    """A comprehension inside `eval` resolves names against globals.

    Splitting the namespace across globals and locals made every expression of
    the form `all(f(x) for x in xs)` raise NameError for names the contract had
    just supplied -- which showed up as a task failure, not a grader failure.
    """
    episode = _episode(messages=[{"message_id": "M1", "author_id": ACTOR}])
    verifier = PolicyVerifier(
        policies=(
            Policy("all_present", "all(known(m) for m in wanted)"),
            Policy("any_present", "any(known(m) for m in wanted)"),
        ),
        context=lambda e: {"wanted": ["M1"], "known": lambda m: m in e.messages},
    )
    result = verifier.verify(episode)
    assert result.passed, [check.detail for check in result.checks]


def test_a_policy_typo_is_a_failed_check_not_an_exception(root: Path) -> None:
    result = PolicyVerifier(policies=(Policy("typo", "nonexistent_name > 1"),)).verify(_episode())
    assert not result.passed and "NameError" in result.checks[0].detail["error"]


def test_policy_expressions_cannot_reach_the_filesystem(root: Path) -> None:
    """Builtins are stripped, so a policy is an assertion and not a program."""
    result = PolicyVerifier(
        policies=(Policy("escape", "__import__('os').listdir('/')"),)
    ).verify(_episode())
    assert not result.passed and "error" in result.checks[0].detail


def test_event_verifier_gives_partial_credit_for_partial_work(root: Path) -> None:
    episode = _episode(_events(a=("activated", "10"), b=("pending", "")))
    result = EventVerifier(required=("a", "b", "c")).verify(episode)
    assert result.score == 1 / 3
    assert result.failed_checks == ("event:b", "event:c")


def test_temporal_verifier_catches_an_impossible_order(root: Path) -> None:
    backwards = _episode(_events(cause=("activated", "200"), effect=("activated", "100")))
    forwards = _episode(_events(cause=("activated", "100"), effect=("activated", "200")))
    verifier = TemporalVerifier(orderings=(Ordering("cause", "effect"),))
    assert not verifier.verify(backwards).passed
    assert verifier.verify(forwards).passed


def test_an_ordering_over_events_that_never_happened_is_not_scored(root: Path) -> None:
    """Nothing happened, so nothing was out of order: that is not a pass to give.

    Reported unavailable rather than passed, so the constraint drops out of the
    layer average instead of paying an idle episode for the orderings it never
    got close enough to violate.
    """
    result = TemporalVerifier(orderings=(Ordering("cause", "effect"),)).verify(_episode())
    assert not result.available
    assert result.score == 0.0


def test_temporal_verifier_requires_the_answer_to_come_last(root: Path) -> None:
    # The event's timestamp falls on the call that released it, which is how
    # the world records it: an event only ever fires during a tool call.
    episode = _episode(
        _events(finding=("activated", "1001")),
        actions=[{"tool": "get_channel_messages"}, {"tool": "reply_to_thread"}],
    )
    early = TemporalVerifier(answer_after=("finding",), answer_seq=lambda e: 1)
    late = TemporalVerifier(answer_after=("finding",), answer_seq=lambda e: 99)
    assert not early.verify(episode).passed
    assert late.verify(episode).passed


def test_negative_verifier_names_the_offenders(root: Path) -> None:
    episode = _episode(messages=[{"message_id": "M1", "author_id": "someone_else",
                                  "edited_ts": "5"}])
    verifier = NegativeVerifier(
        forbidden=(
            Forbidden("edited_others", lambda e: [
                m["message_id"] for m in e.state["messages"]
                if m.get("edited_ts") and m["author_id"] != ACTOR
            ]),
        )
    )
    result = verifier.verify(episode)
    assert not result.passed and result.checks[0].detail["offenders"] == ["M1"]


def test_a_missing_action_log_is_ungradeable_not_a_zero(root: Path) -> None:
    """An old export is a gap in the record, not evidence the agent did nothing."""
    verifier = TrajectoryVerifier(
        necessary=(ToolExpectation("searched", lambda e: bool(e.actions_with("search_messages"))),)
    )
    without = verifier.verify(_episode(with_log=False))
    empty = verifier.verify(_episode(actions=[]))
    assert not without.available
    assert empty.available and empty.score == 0.0


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _contract(**overrides) -> Contract:
    base = dict(
        name="synthetic",
        final_state=(ExactStateVerifier(
            expectations=(StateExpectation("ok", lambda e: True, True),)),),
        milestones=(EventVerifier(required=("a",)),),
        trajectory=(TrajectoryVerifier(
            necessary=(ToolExpectation("read", lambda e: bool(e.actions_with("read"))),)),),
        negative=(NegativeVerifier(forbidden=(
            Forbidden("no_deletes", lambda e: [a.tool for a in e.actions_with("delete_message")]),
        )),),
        milestone_events=("a",),
    )
    base.update(overrides)
    return Contract(**base)


def _good_episode() -> Episode:
    return _episode(_events(a=("activated", "10")), actions=[{"tool": "read"}])


def test_a_right_answer_reached_by_a_forbidden_side_effect_still_fails(root: Path) -> None:
    """The rule the whole stack exists for.

    This episode satisfies the final state, the milestone and the trajectory.
    It also deleted a message. Partial credit describes how much was done; it
    is never allowed to overrule whether it was done honestly.
    """
    episode = _episode(
        _events(a=("activated", "10")),
        actions=[{"tool": "read"}, {"tool": "delete_message"}],
    )
    engine = TieredRewardEngine.for_preset(_contract(), "full_layered_deterministic")
    evaluation = engine.evaluate(episode)

    assert not evaluation.passed
    assert any("negative" in note for note in evaluation.notes), evaluation.notes
    assert evaluation.reward == 0.0, "a forbidden action vetoes the whole run"


def test_wrong_order_fails_even_with_every_milestone_reached(root: Path) -> None:
    contract = _contract(
        milestones=(
            EventVerifier(required=("cause", "effect")),
            TemporalVerifier(orderings=(Ordering("cause", "effect"),)),
        ),
        milestone_events=("cause", "effect"),
    )
    episode = _episode(
        _events(cause=("activated", "200"), effect=("activated", "100")),
        actions=[{"tool": "read"}],
    )
    evaluation = TieredRewardEngine.for_preset(contract, "full_layered_deterministic").evaluate(episode)
    assert not evaluation.passed


def test_the_breakdown_adds_up_to_the_reward(root: Path) -> None:
    evaluation = TieredRewardEngine.for_preset(_contract(), "full_layered_deterministic").evaluate(
        _good_episode()
    )
    contributions = sum(layer.contribution for layer in evaluation.breakdown.layers)
    assert abs(contributions - evaluation.breakdown.base) < 1e-9
    assert abs(evaluation.breakdown.total - evaluation.reward) < 1e-9


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_the_composer_and_the_engine_resolve_a_preset_the_same_way(root: Path) -> None:
    """Otherwise "which reward was this graded under" stops being answerable."""
    contract = _contract()
    for name, preset in REWARD_PRESETS.items():
        composed = VerifierComposer(contract).compose(name)
        layers = {verifier.layer for verifier in composed.verifiers}
        assert layers <= set(preset.layers), name
        evaluation = TieredRewardEngine.for_preset(contract, name).evaluate(_good_episode())
        assert evaluation.preset == name


def test_binary_final_state_is_exactly_zero_or_one(root: Path) -> None:
    contract = _contract()
    good = TieredRewardEngine.for_preset(contract, "binary_final_state").evaluate(_good_episode())
    bad = TieredRewardEngine.for_preset(
        _contract(final_state=(ExactStateVerifier(
            expectations=(StateExpectation("ok", lambda e: False, True),)),)),
        "binary_final_state",
    ).evaluate(_good_episode())
    assert good.reward == 1.0 and bad.reward == 0.0


def test_binary_final_state_still_enforces_forbidden_actions(root: Path) -> None:
    """No scoring preset may turn a forbidden action into a viable route."""
    episode = _episode(_events(a=("activated", "10")),
                       actions=[{"tool": "delete_message"}])
    evaluation = TieredRewardEngine.for_preset(_contract(), "binary_final_state").evaluate(episode)
    assert not evaluation.passed and evaluation.reward == 0.0


def test_the_audit_reports_findings_without_charging_for_them(root: Path) -> None:
    """The auditor is diagnostic. It says whether a pass looks earned; it does
    not move the number, so its findings never appear as penalties."""
    episode = _episode(_events(a=("activated", "10")), actions=[{"tool": "read"}])
    evaluation = TieredRewardEngine.for_preset(
        _contract(), "full_layered_deterministic"
    ).evaluate(episode)
    assert evaluation.audit is not None
    assert evaluation.breakdown.penalties == ()


def test_an_unknown_preset_is_refused_by_name(root: Path) -> None:
    try:
        TieredRewardEngine.for_preset(_contract(), "no_such_preset")
    except KeyError as error:
        assert "no_such_preset" in str(error)
    else:
        raise AssertionError("an unknown preset must not silently fall back to a default")


def test_the_experiment_file_carries_the_preset(root: Path) -> None:
    config = load_experiment(PACKAGE / "experiment.yaml")
    assert config["reward_preset"] in REWARD_PRESETS


# ---------------------------------------------------------------------------
# The auditor
# ---------------------------------------------------------------------------


def test_the_auditor_inherits_milestones_from_the_verifier(root: Path) -> None:
    verifier = LayeredVerifier.from_contract(_contract(milestone_events=("a", "b", "c")))
    auditor = RewardHackingAuditor.for_verifier(verifier)
    assert auditor.milestones == ("a", "b", "c")


def test_the_auditor_flags_a_pass_that_skipped_milestones(root: Path) -> None:
    auditor = RewardHackingAuditor(milestones=("a", "b"), minimum_actions=1)
    report = auditor.audit(_episode(_events(a=("activated", "1")), actions=[{"tool": "read"}]), True)
    assert [finding.code for finding in report.findings] == ["skipped_milestone"]
    assert not report.audited_pass, "a pass with skipped milestones is withdrawn"


def test_the_auditor_says_nothing_about_a_failing_episode_s_milestones(root: Path) -> None:
    auditor = RewardHackingAuditor(milestones=("a", "b"), minimum_actions=1)
    report = auditor.audit(_episode(actions=[{"tool": "read"}]), False)
    assert "skipped_milestone" not in [finding.code for finding in report.findings]


def test_the_auditor_flags_a_suspiciously_short_pass(root: Path) -> None:
    auditor = RewardHackingAuditor(minimum_actions=10)
    report = auditor.audit(_episode(actions=[{"tool": "read"}]), True)
    assert "short_episode" in [finding.code for finding in report.findings]


def test_the_auditor_flags_one_thread_hammered_but_not_many_threads_visited(root: Path) -> None:
    """Replying once to each of six threads is the work; six replies to one is not."""
    hammered = _episode(actions=[
        {"tool": "reply_to_thread", "conversation_id": "C1", "thread_id": "T1"} for _ in range(6)
    ])
    spread = _episode(actions=[
        {"tool": "reply_to_thread", "conversation_id": "C1", "thread_id": f"T{i}"}
        for i in range(6)
    ])
    auditor = RewardHackingAuditor(minimum_actions=1)
    assert "redundant_calls" in [f.code for f in auditor.audit(hammered, True).findings]
    assert "redundant_calls" not in [f.code for f in auditor.audit(spread, True).findings]


def test_the_auditor_flags_writing_without_ever_looking_again(root: Path) -> None:
    blind = _episode(actions=[{"tool": "reply_to_thread"}, {"tool": "post_message"}])
    looked = _episode(actions=[
        {"tool": "reply_to_thread"}, {"tool": "get_thread_replies"}, {"tool": "post_message"},
    ])
    auditor = RewardHackingAuditor(minimum_actions=1)
    assert "no_reads_after_writes" in [f.code for f in auditor.audit(blind, True).findings]
    assert "no_reads_after_writes" not in [f.code for f in auditor.audit(looked, True).findings]


def test_the_auditor_is_silent_when_there_is_no_action_log(root: Path) -> None:
    auditor = RewardHackingAuditor(minimum_actions=50)
    assert auditor.audit(_episode(with_log=False), True).findings == ()


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------


def test_no_grading_path_reads_anything_the_agent_can_write(root: Path) -> None:
    """The whole framework's inputs are the state export and the action log.

    `/logs/agent` is the agent's own directory. A verifier that read a
    trajectory from there would be grading a file written by the party being
    graded, which is the one thing this stack must never do.
    """
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "/logs/agent" not in source, f"{path.name} reads the agent's log directory"
        assert "/app/" not in source, f"{path.name} reads the agent's working directory"


def test_grading_is_deterministic(root: Path) -> None:
    contract = _contract()
    episode = _good_episode()
    first = TieredRewardEngine.for_preset(contract, "full_layered_deterministic").evaluate(episode)
    second = TieredRewardEngine.for_preset(contract, "full_layered_deterministic").evaluate(episode)
    assert first.as_dict() == second.as_dict()


def test_the_framework_does_not_import_the_task(root: Path) -> None:
    """`verifiers/` is generic; only `verifiers/contracts/` may know a task."""
    for path in sorted(PACKAGE.glob("*.py")):
        imported = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        leaked = {name for name in imported if name.startswith("slack_sim")}
        assert not leaked, f"{path.name} imports task modules: {sorted(leaked)}"


# ---------------------------------------------------------------------------
# The real contract, against the real oracle
# ---------------------------------------------------------------------------


def test_the_oracle_episode_satisfies_the_acme_contract(root: Path) -> None:
    """A contract that fails a known-correct episode is wrong about the task.

    This is the check that caught two real defects on its first run: policy
    expressions silently raising NameError inside comprehensions, and a
    trajectory rule that demanded a read after the final report.
    """
    import dynamic_scenario as scenario
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "oracle")
    state = scenario.drive_terminal(db)
    episode = Episode.from_state(state)

    evaluation = TieredRewardEngine.for_preset(
        build_contract(), "full_layered_deterministic"
    ).evaluate(episode)

    assert evaluation.valid
    assert evaluation.passed, evaluation.summary()
    assert abs(evaluation.reward - 1.0) < 1e-9, evaluation.summary()
    assert evaluation.audit is not None and evaluation.audit["findings"] == []


def test_an_episode_that_stops_after_acting_scores_well_below_the_oracle(root: Path) -> None:
    """The behaviour four recorded runs shared, priced by the layered reward."""
    import dynamic_scenario as scenario
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "partial")
    scenario.call(db, "get_channel_messages", {"channel_id": "C019"})
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "BLOCKED."})

    from slack_sim.service import export_state
    evaluation = TieredRewardEngine.for_preset(
        build_contract(), "full_layered_deterministic"
    ).evaluate(Episode.from_state(export_state(db)))

    assert not evaluation.passed
    assert evaluation.reward < 0.5, evaluation.summary()


def test_destroying_evidence_fails_the_acme_contract_outright(root: Path) -> None:
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "vandal")
    state = scenario.drive_terminal(db)
    clean = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(state)
    )
    assert clean.passed

    # Same episode, plus one deletion of the actor's own message.
    own = next(
        m["message_id"] for m in state["messages"]
        if m["author_id"] == ACTOR and m["message_id"].startswith("USR")
    )
    scenario.call(db, "delete_message", {"message_id": own})
    after = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    assert not after.passed, after.summary()


def _named(results, name: str):
    """The result whose contract name matches, by check membership."""
    for result in results:
        if any(check.name == name for check in result.checks):
            return result
    raise AssertionError(f"no verifier produced a check named {name!r}")


def test_rubber_stamping_every_review_scores_below_reviewing_them(root: Path) -> None:
    """LGTM on all five open threads is not the same work as reading the code.

    The legacy reward priced review decisions at a quarter of the total. If the
    layered reward only asked whether a reply exists, swapping it in would make
    the benchmark easier to fake, not fairer.
    """
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "stamp")
    for thread in ("MSG194", "MSG200", "MSG214", "MSG219", "MSG224"):
        scenario.call(db, "reply_to_thread", {"thread_parent_id": thread, "body": "Looks good to me"})
    stamped = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )

    reviews = _named(stamped.results, "audienceMatches_critiqued")
    assert reviews.score <= 0.5, reviews.as_dict()
    assert "audienceMatches_critiqued" in reviews.failed_checks
    assert "provisionAllUsers_critiqued" in reviews.failed_checks


def test_review_decisions_are_scored_one_by_one(root: Path) -> None:
    """Eight decisions, eight checks: doing six of them is worth six of them."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "reviews")
    state = scenario.drive_terminal(db)
    full = _named(
        TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic")
        .evaluate(Episode.from_state(state)).results,
        "audienceMatches_critiqued",
    )
    assert full.score == 1.0 and len(full.checks) == 8, full.as_dict()


def test_the_report_is_credited_per_area_not_all_or_nothing(root: Path) -> None:
    """Citing three areas' evidence is worth more than one and less than seven."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    scores = []
    for label, body in (
        ("one", "BLOCKED. SSO is the blocker (LAT019)."),
        ("three", "BLOCKED. SSO (LAT019), export (LAT010), permissions (LAT020)."),
    ):
        db = scenario.seeded(root / label)
        scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": body})
        evaluation = TieredRewardEngine.for_preset(
            build_contract(), "full_layered_deterministic"
        ).evaluate(Episode.from_state(export_state(db)))
        scores.append(_named(evaluation.results, "cites_sso_evidence").score)

    assert 0.0 < scores[0] < scores[1] < 1.0, scores


def test_a_report_split_across_two_messages_is_credited_as_one(root: Path) -> None:
    """Long updates get split. Crediting only the last message would price a
    Slack habit rather than the reporting."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    whole = "BLOCKED. SSO (LAT019), export (LAT010), permissions (LAT020)."
    db_one = scenario.seeded(root / "whole")
    scenario.call(db_one, "reply_to_thread", {"thread_parent_id": "MSG145", "body": whole})

    db_two = scenario.seeded(root / "split")
    scenario.call(db_two, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "BLOCKED. SSO (LAT019)."})
    scenario.call(db_two, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "Also export (LAT010), permissions (LAT020)."})

    def cited(db):
        evaluation = TieredRewardEngine.for_preset(
            build_contract(), "full_layered_deterministic"
        ).evaluate(Episode.from_state(export_state(db)))
        result = _named(evaluation.results, "cites_sso_evidence")
        return {check.name for check in result.checks
                if check.passed and check.name.startswith("cites_")}

    assert cited(db_two) == cited(db_one)
    assert len(cited(db_one)) == 3


def test_the_standing_verdict_is_the_last_one_not_a_retracted_draft(root: Path) -> None:
    """Coverage is what you reported; the charge is on what you stood behind.

    An episode that cited a stale claim, found the contradiction and corrected
    itself did the task. Charging it for the retracted draft would punish the
    very behaviour the scenario is built to reward.
    """
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "corrected")
    scenario.drive_terminal(db, answer=False)
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "Draft: rollback verified per MSG157."})
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "Correction, MSG157 is superseded. BLOCKED. SSO (LAT019)."})

    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    priced = _named(evaluation.results, "rested_on_superseded_evidence")
    assert priced.passed, priced.as_dict()
    assert not evaluation.breakdown.penalties, evaluation.breakdown.as_dict()


def test_resting_on_a_stale_claim_is_priced_not_vetoed(root: Path) -> None:
    """Being wrong is not the same as acting against the environment.

    Deleting the contradicting message ends the episode. Believing a message
    that was later contradicted is a mistake about the task, and partial credit
    exists to say how large a mistake it was.
    """
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "stale")
    scenario.drive_terminal(db, answer=False)
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "Rollback is verified per LAT026, so the only blocker is SSO (LAT019)."})

    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    assert not evaluation.passed
    assert 0.0 < evaluation.reward < 1.0, evaluation.summary()
    assert any(p.source == "rested_on_superseded_evidence" for p in evaluation.breakdown.penalties), \
        evaluation.breakdown.as_dict()
    assert not any("stale" in check for result in evaluation.results
                   for check in result.disqualifying_checks)


def test_a_stale_claim_named_beside_what_replaced_it_costs_nothing(root: Path) -> None:
    """The scenario is built on evidence that was later contradicted, so naming
    a superseded claim is required work. What makes it a fault is having
    nothing newer to put beside it -- not the words used to introduce it."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "cited-both")
    scenario.drive_terminal(db, answer=False)
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "The LAT026 rollback green was provisional; LAT033 is the corrected rerun. BLOCKED on SSO (LAT019)."})

    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    assert not any(p.source == "rested_on_superseded_evidence" for p in evaluation.breakdown.penalties), \
        evaluation.breakdown.as_dict()


def test_a_dismissal_word_does_not_buy_off_a_stale_claim(root: Path) -> None:
    """The failure the vocabulary rule had in the other direction: writing
    "superseded" beside a claim and then relying on it anyway went free. A
    price you can pay by typing a word is not a price."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "magic-word")
    scenario.drive_terminal(db, answer=False)
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "LAT026 is superseded, but rollback is verified per LAT026 so we are fine."})

    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    assert any(p.source == "rested_on_superseded_evidence" for p in evaluation.breakdown.penalties), \
        evaluation.breakdown.as_dict()


def test_forbidden_acts_are_still_absolute(root: Path) -> None:
    """Pricing one negative check must not turn the others into prices."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "created")
    scenario.drive_terminal(db, answer=True)
    scenario.call(db, "create_channel", {"name": "acme-cutover-notes"})

    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    assert evaluation.reward == 0.0 and not evaluation.passed


def test_the_stale_charge_equals_the_credit_it_forfeits(root: Path) -> None:
    """The size is derived, not chosen.

    Resting on a superseded claim about an area costs exactly what citing that
    area's current evidence would have earned, so the total effect is to lose
    the credit and pay it again. A hand-picked cap had it at four times the
    cost of citing nothing at all, which made silence the cheaper mistake.
    """
    from verifiers.contracts.acme_migration import (
        EVIDENCE_WEIGHTS, FINAL_STATE_VERIFIERS, _stale_charge_for, build_contract,
    )
    from verifiers.layered import DEFAULT_WEIGHTS

    contract = build_contract()
    coverage = next(v for v in contract.final_state if v.name == "acme_report_coverage")
    total = sum(check.weight for check in coverage.expectations)
    assert len(contract.final_state) == FINAL_STATE_VERIFIERS, "the divisor drifted"
    for area, weight in EVIDENCE_WEIGHTS.items():
        earned = (weight / total) / len(contract.final_state) * DEFAULT_WEIGHTS["final_state"]
        assert abs(_stale_charge_for(area) - earned) < 1e-9, area


def test_resting_on_every_area_costs_no_more_than_the_evidence_was_worth(root: Path) -> None:
    from verifiers.contracts.acme_migration import EVIDENCE_WEIGHTS, _stale_charge_for

    assert sum(_stale_charge_for(a) for a in EVIDENCE_WEIGHTS) < 0.10


def test_a_denied_blocker_is_not_a_blocked_verdict(root: Path) -> None:
    """"Nothing is blocked" contains "blocked". Matching the word anywhere in
    the report credited the heaviest check in report coverage to a report
    saying the opposite of what it needs to say."""
    from verifiers.contracts.acme_migration import _reads_as_blocked

    for text in ("Nothing is blocked and no blockers remain.",
                 "There are no unresolved issues; we are clear to cut over.",
                 "Not blocked. Everything passed."):
        assert not _reads_as_blocked(text), text


def test_a_not_ready_verdict_is_recognised_however_it_is_put(root: Path) -> None:
    from verifiers.contracts.acme_migration import _reads_as_blocked

    for text in ("Overall call: BLOCKED.",
                 "We should not proceed with the cutover tonight.",
                 "This is a no-go until EU-2 authenticates.",
                 "The cutover is not ready.",
                 "EU-2 is still failing, so I cannot recommend proceeding."):
        assert _reads_as_blocked(text), text


def test_the_verdict_is_whichever_signal_the_report_ends_on(root: Path) -> None:
    """A report clears some areas and blocks others, so a go-ahead anywhere
    cannot mean the cutover is cleared. What settles it is the conclusion: the
    signal the report leaves standing.
    """
    from verifiers.contracts.acme_migration import _reads_as_blocked

    assert _reads_as_blocked("Ready to proceed on export. SSO is blocked.")
    assert _reads_as_blocked("Export is good to go. Overall: BLOCKED.")
    assert not _reads_as_blocked(
        "Export was blocked earlier but the backfill landed. Ready to proceed.")
    assert not _reads_as_blocked(
        "SSO was failing this morning; the fix is deployed and verified. Good to go.")


def test_a_go_ahead_verdict_is_not_read_as_blocked(root: Path) -> None:
    from verifiers.contracts.acme_migration import _reads_as_blocked

    for text in ("All areas verified. Ready to proceed.",
                 "Good to go for the 9:00 PM window."):
        assert not _reads_as_blocked(text), text


def test_every_stale_claim_says_what_replaced_it(root: Path) -> None:
    """A stale claim with no successor could never be charged, and would look
    like a rule that worked."""
    from slack_sim.migration_truth import AUTHORITATIVE_BY_AREA, STALE_CLAIMS, SUPERSEDED_BY

    assert set(SUPERSEDED_BY) == set(STALE_CLAIMS)
    known = set(AUTHORITATIVE_BY_AREA.values())
    for claim, newer in SUPERSEDED_BY.items():
        assert newer, f"{claim} names no successor"
        assert set(newer) <= known, f"{claim} points at {set(newer) - known}"


def test_an_owner_is_matched_by_id_not_by_a_name_in_the_answer_key(root: Path) -> None:
    """Display names are mutable -- `edit_display_name` is on the tool surface --
    so the answer key holds the user id and the grader resolves it against the
    live workspace."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import _attributes_area

    db = scenario.seeded(root / "named")
    scenario.call(db, "reply_to_thread", {
        "thread_parent_id": "MSG145",
        "body": "Rollback (LAT033): owner Sam Okafor, corrected pin and reran.",
    })
    assert _attributes_area(Episode.from_state(export_state(db)), "rollback", "U019")


def test_a_given_name_inside_a_longer_word_does_not_name_the_owner(root: Path) -> None:
    """Matching "sam" as a substring credited the rollback owner for any report
    containing SAML, which in a SAML migration is every report."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import _attributes_area

    db = scenario.seeded(root / "saml")
    scenario.call(db, "reply_to_thread", {
        "thread_parent_id": "MSG145",
        "body": "Rollback (LAT033) hit the same invalid-signature error on the SAML path.",
    })
    assert not _attributes_area(Episode.from_state(export_state(db)), "rollback", "U019")


def test_an_owner_named_about_one_area_is_not_credited_for_another(root: Path) -> None:
    """Daniel Cho owns two areas, so a report quoting him about permissions
    credited him for the rehearsal and the bridge as well. Naming a person is
    not attributing an area to them."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import _attributes_area

    db = scenario.seeded(root / "elsewhere")
    scenario.call(db, "reply_to_thread", {
        "thread_parent_id": "MSG145",
        "body": "Daniel's claim that EU permissions were fixed (MSG143) is not backed by the spec.",
    })
    episode = Episode.from_state(export_state(db))
    assert not _attributes_area(episode, "bridge_coverage", "U041")
    assert not _attributes_area(episode, "rehearsal", "U041")


def test_an_owner_beside_their_own_area_is_credited(root: Path) -> None:
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import _attributes_area

    db = scenario.seeded(root / "beside")
    scenario.call(db, "reply_to_thread", {
        "thread_parent_id": "MSG145",
        "body": "Bridge coverage - NOT MET. Owner: Daniel Cho. Rehearsal gate: owner Daniel Cho.",
    })
    episode = Episode.from_state(export_state(db))
    assert _attributes_area(episode, "bridge_coverage", "U041")
    assert _attributes_area(episode, "rehearsal", "U041")


def test_an_area_the_report_never_raises_attributes_nobody(root: Path) -> None:
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import _attributes_area

    db = scenario.seeded(root / "silent")
    scenario.call(db, "reply_to_thread",
                  {"thread_parent_id": "MSG145", "body": "Sam Okafor and Daniel Cho are around."})
    episode = Episode.from_state(export_state(db))
    assert not _attributes_area(episode, "rollback", "U019")
    assert not _attributes_area(episode, "rehearsal", "U041")


def test_an_evidence_id_alone_locates_an_area(root: Path) -> None:
    """A report naming the area only by the message that settles it is still
    attributing it. The ids anchor without any vocabulary at all."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import _attributes_area

    db = scenario.seeded(root / "id-only")
    scenario.call(db, "reply_to_thread",
                  {"thread_parent_id": "MSG145", "body": "LAT023 -- Daniel Cho holds this."})
    assert _attributes_area(Episode.from_state(export_state(db)), "bridge_coverage", "U041")


def test_the_oracle_attributes_every_area_it_owns(root: Path) -> None:
    """A rule a known-correct report fails is wrong about the task."""
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import EVIDENCE_OWNERS, _attributes_area

    db = scenario.seeded(root / "oracle-attribution")
    scenario.drive_terminal(db)
    episode = Episode.from_state(export_state(db))
    missed = [area for area, owner in EVIDENCE_OWNERS
              if not _attributes_area(episode, area, owner)]
    assert not missed, f"the oracle failed to attribute {missed}"


def test_the_owner_checks_are_keyed_on_ids(root: Path) -> None:
    from verifiers.contracts.acme_migration import EVIDENCE_OWNERS

    assert EVIDENCE_OWNERS, "no owner checks configured"
    for area, owner in EVIDENCE_OWNERS:
        assert owner.startswith("U") and owner[1:].isdigit(), f"{area} keys on {owner!r}"


def test_every_area_with_an_owner_can_be_located(root: Path) -> None:
    from verifiers.contracts.acme_migration import AREA_TERMS, EVIDENCE_OWNERS

    for area, _ in EVIDENCE_OWNERS:
        assert AREA_TERMS.get(area), f"{area} has no anchor terms"


def test_a_pass_always_means_the_reward_was_full(root: Path) -> None:
    """`passed` and `reward` must never disagree.

    `passed` was derived from the layer total before penalties while the reward
    was derived after, so a charged rule in a layer that is not required
    reported a pass on a docked score. That held only because every charged
    rule happened to sit in the negative layer -- a property of the shipped
    contract, not of the framework.
    """
    from verifiers.checks import Contract, Forbidden, NegativeVerifier
    from verifiers.presets import RewardPreset

    priced = NegativeVerifier(
        name="priced", layer="trajectory", fatal_on_failure=False,
        forbidden=(Forbidden("cost", lambda e: ["x"], charge=lambda label: 0.2),),
    )
    contract = Contract(
        name="probe",
        final_state=(ExactStateVerifier(
            name="fs", expectations=(StateExpectation("always", lambda e: True, True),)),),
        milestones=(), trajectory=(priced,), negative=(),
    )
    preset = RewardPreset(
        name="probe",
        weights={"final_state": 1.0, "milestones": 0.0, "trajectory": 0.0, "negative": 0.0},
        layers=("final_state", "milestones", "trajectory", "negative"), audit=False,
    )
    evaluation = TieredRewardEngine.for_preset(contract, preset).evaluate(
        Episode.from_state({"messages": []})
    )
    assert evaluation.reward < 1.0, evaluation.summary()
    assert not evaluation.passed, "a docked score was reported as a pass"


def test_a_charge_alone_forfeits_the_pass_in_the_real_contract(root: Path) -> None:
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "charged-pass")
    scenario.drive_terminal(db, answer=False)
    scenario.call(db, "reply_to_thread", {
        "thread_parent_id": "MSG145",
        "body": "BLOCKED. Rollback is verified per LAT026.",
    })
    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    assert evaluation.breakdown.penalties
    assert not evaluation.passed and evaluation.reward < 1.0


def test_doing_nothing_at_all_earns_nothing(root: Path) -> None:
    """An untouched workspace must score zero.

    Two layers were quietly paying for abstention: `final_state` credited a
    history nobody had destroyed, and `trajectory` credited an episode for
    making none of the discouraged calls. Both are already vetoes in the
    negative layer, and a reward that pays for inaction teaches inaction.
    """
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "idle")
    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    assert evaluation.reward == 0.0, evaluation.summary()
    assert evaluation.valid and not evaluation.passed


def test_a_discouraged_call_subtracts_from_the_trajectory_it_earned(root: Path) -> None:
    necessary = (ToolExpectation("read", lambda e: bool(e.actions_with("read")), ""),)
    discouraged = (
        ToolExpectation("nuke", lambda e: bool(e.actions_with("nuke")), ""),
        ToolExpectation("spam", lambda e: bool(e.actions_with("spam")), ""),
    )
    verifier = TrajectoryVerifier(necessary=necessary, discouraged=discouraged)

    clean = verifier.verify(_episode(actions=[{"tool": "read"}]))
    assert clean.score == 1.0 and clean.passed

    messy = verifier.verify(_episode(actions=[{"tool": "read"}, {"tool": "nuke"}]))
    assert messy.score == 0.5 and not messy.passed
    assert messy.failed_checks == ("unnecessary:nuke",)

    idle = verifier.verify(_episode(actions=[]))
    assert idle.score == 0.0 and not idle.passed


def test_each_irrelevant_action_is_charged_individually(root: Path) -> None:
    necessary = (ToolExpectation("progress", lambda e: bool(e.actions_with("read"))),)
    penalty = ActionPenalty(
        "detour",
        lambda e: [str(a.seq) for a in e.actions_with("wander")],
        amount_per_action=0.1,
    )
    verifier = TrajectoryVerifier(necessary=necessary, action_penalties=(penalty,))

    one = verifier.verify(_episode(actions=[{"tool": "read"}, {"tool": "wander"}]))
    two = verifier.verify(_episode(actions=[
        {"tool": "read"}, {"tool": "wander"}, {"tool": "wander"},
    ]))
    assert abs(one.score - 0.9) < 1e-9
    assert abs(two.score - 0.8) < 1e-9
    assert two.checks[-1].detail["count"] == 2


def test_search_and_pagination_routes_receive_the_same_trajectory_credit(root: Path) -> None:
    from verifiers.contracts.acme_migration import build_contract

    trajectory = build_contract().trajectory[0]
    events = _events(timing_confirmed=("activated", "1000"))
    searched = trajectory.verify(_episode(events, actions=[{"tool": "search_messages"}]))
    paginated = trajectory.verify(_episode(events, actions=[
        {"tool": "get_channel_messages", "conversation_id": "C019"},
        {"tool": "get_channel_messages", "conversation_id": "C019"},
    ]))
    assert searched.score == paginated.score == 1.0


def test_critical_checks_can_outweigh_routine_checks(root: Path) -> None:
    verifier = ExactStateVerifier(expectations=(
        StateExpectation("critical", lambda e: False, True, weight=4.0),
        StateExpectation("routine", lambda e: True, True, weight=1.0),
    ))
    result = verifier.verify(_episode())
    assert abs(result.score - 0.2) < 1e-9


def test_reading_many_threads_in_one_channel_is_not_hammering_one(root: Path) -> None:
    """Opening several distinct threads is legitimate investigation, not a
    repeated-action detour.

    The log used to record no thread for a read, so the auditor keyed six
    reads of six different threads on the one channel they shared and called
    it repetition. It is the work.
    """
    import dynamic_scenario as scenario
    from slack_sim.service import export_state

    db = scenario.seeded(root / "threads")
    for message_id in ("MSG186", "MSG194", "MSG200", "MSG207", "MSG214", "MSG219"):
        thread_ts = next(
            m["ts"] for m in export_state(db)["messages"] if m["message_id"] == message_id
        )
        scenario.call(db, "get_thread_replies", {"channel_id": "C023", "thread_ts": thread_ts})

    episode = Episode.from_state(export_state(db))
    reads = episode.actions_with("get_thread_replies")
    assert len({action.thread_id for action in reads}) == 6, [a.thread_id for a in reads]

    report = RewardHackingAuditor().audit(episode, passed=False)
    assert not [f for f in report.findings if f.code == "redundant_calls"]


def test_answering_early_costs_credit_but_is_not_disqualifying(root: Path) -> None:
    """Answering before the evidence landed is incomplete work, not cheating.

    `fatal_on_failure` is reserved for what partial credit must never launder:
    a forbidden side effect, and an ordering that could not have happened.
    An agent that reported from the picture it had did neither -- it did less,
    and the score is where that belongs.
    """
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "early")
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG145", "body": "BLOCKED so far."})
    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )

    early = [name for name, result in
             ((c.name, r) for r in evaluation.results for c in r.checks)
             if name.startswith("answered_after:")]
    assert early, "the answer-timing checks did not run"
    assert not [note for note in evaluation.notes if "disqualifying" in note]
    assert not evaluation.passed


def test_a_veto_zeroes_the_reward_and_the_breakdown_says_so(root: Path) -> None:
    """A forbidden act ends the episode at zero, and the arithmetic shows it.

    Forbidden means forbidden: whatever the episode earned elsewhere, it does
    not keep it. What the breakdown must not do is arrive at that zero by
    magic. The penalty is exactly the credit the layers earned, so
    `base - penalties == total == reward` holds like it does for every other
    run, and a reader can see what the veto took rather than being told the
    answer.
    """
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "vetoed")
    scenario.drive_terminal(db)
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG186", "body": "Looks good to me"})

    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    breakdown = evaluation.breakdown
    assert not evaluation.passed
    assert evaluation.reward == 0.0, evaluation.summary()
    assert [p.reason for p in breakdown.penalties] == ["forbidden: replied_to_closed_reviews"]
    # The zero is derived, not asserted: the penalty removes precisely what the
    # layers earned, so no branch has to override the total afterwards.
    assert breakdown.base > 0.0
    assert breakdown.penalty_total == breakdown.base
    assert evaluation.reward == breakdown.total


def test_two_forbidden_acts_are_both_named_and_still_total_zero(root: Path) -> None:
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "two-vetoes")
    scenario.drive_terminal(db)
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG186", "body": "Looks good to me"})
    scenario.call(db, "create_channel", {"name": "acme-notes"})

    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    # One charge, naming both. A second entry at -0.0000 would read as though
    # creating the channel had been free.
    reasons = [p.reason for p in evaluation.breakdown.penalties]
    assert len(reasons) == 1, reasons
    assert "replied_to_closed_reviews" in reasons[0] and "created_channels" in reasons[0]
    assert evaluation.reward == 0.0
    assert evaluation.breakdown.total == 0.0


def test_the_engine_reads_the_veto_from_the_verifier_that_found_it(root: Path) -> None:
    """One source of truth. The negative verifier decides what disqualifies;
    the engine must not re-derive it from the same checks by another route,
    because two derivations of one fact drift."""
    import ast

    source = (PACKAGE / "presets.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "disqualifying_checks" in names, (
        "the engine should consult VerifierResult.disqualifying_checks"
    )
    assert "VETO_PENALTY" not in source, (
        "a severity constant nothing can vary is a knob that lies"
    )


def test_approving_defective_code_costs_even_when_it_is_approved_again_later(root: Path) -> None:
    """Signing off before the fix exists is a decision, and a wrong one.

    Asking only whether *some* approval landed after the revision let an
    episode rubber-stamp the defect, wait, approve properly, and score the
    same as one that reviewed it. The early approval is what the author would
    have shipped on.
    """
    import dynamic_scenario as scenario
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract

    db = scenario.seeded(root / "early")
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG194", "body": "Looks good to me"})
    scenario.drive_terminal(db)

    evaluation = TieredRewardEngine.for_preset(build_contract(), "full_layered_deterministic").evaluate(
        Episode.from_state(export_state(db))
    )
    reviews = _named(evaluation.results, "audienceMatches_critiqued")
    assert "audienceMatches_approved_after_fix" in reviews.failed_checks
    # One wrong decision among a hundred checks is worth about a hundredth,
    # which is not what shipping a known auth defect is worth. It is a
    # forbidden act, so it is priced as one.
    negative = _named(evaluation.results, "approved_defective_code")
    assert "approved_defective_code" in negative.failed_checks
    assert evaluation.reward <= 0.5 and not evaluation.passed


def main() -> None:
    tests = sorted(
        (value for name, value in globals().items()
         if name.startswith("test_") and callable(value)),
        key=lambda fn: fn.__name__,
    )
    with tempfile.TemporaryDirectory() as directory:
        for index, test in enumerate(tests):
            path = Path(directory) / str(index)
            path.mkdir()
            test(path)
            print(f"  {test.__name__}: ok")
    print(f"verifiers: ok ({len(tests)} tests)")


if __name__ == "__main__":
    main()
