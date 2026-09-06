#!/usr/bin/env python3
"""Does the reward mean what it claims to mean?

Two realistic episodes, graded end to end through the same entry point Harbor
calls. One is a competent run written independently of the oracle: different
order, different retrieval routes, the timing question asked by DM, the update
split across two messages. The other is a capable agent that stopped short,
modelled on the recorded Opus 4.7 runs.

The pair is the check. A reward that only pays the script it was written
against is measuring the script; a reward that cannot separate "did less" from
"cheated" is measuring nothing useful. So the competent run must be paid in
full, the short run must be paid partially, and every check the short run
failed must be one the competent run passed -- because a check nobody can pass
is a bug in the grader, not a finding about the agent.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reward_episodes as episodes
import test_migration_readiness as entry
from verifiers import Episode, RewardHackingAuditor, TieredRewardEngine
from verifiers.contracts.acme_migration import build_contract


def _evaluate(state, preset="full_layered_deterministic"):
    return TieredRewardEngine.for_preset(build_contract(), preset).evaluate(
        Episode.from_state(state)
    )


def _checks(evaluation):
    """Every check that actually ran. A layer reported unavailable did not run,
    so its placeholder is a grading gap rather than a failed check."""
    return {check.name: check.passed
            for result in evaluation.results if result.available
            for check in result.checks}


# ---------------------------------------------------------------------------
# The competent run
# ---------------------------------------------------------------------------


def test_a_correct_run_that_is_not_the_oracle_is_still_paid_in_full(root: Path) -> None:
    state = episodes.competent_run(episodes.seeded(root / "competent"))
    evaluation = _evaluate(state)

    assert evaluation.valid and evaluation.passed, evaluation.summary()
    assert evaluation.reward == 1.0, evaluation.summary()
    for layer in evaluation.breakdown.layers:
        assert layer.available and layer.score == 1.0, evaluation.summary()


def test_the_competent_run_survives_the_audit(root: Path) -> None:
    state = episodes.competent_run(episodes.seeded(root / "audited"))
    evaluation = _evaluate(state)
    assert evaluation.audit is not None
    assert evaluation.audit["findings"] == [], evaluation.audit
    assert evaluation.breakdown.penalty_total == 0.0


def test_every_verifier_type_gets_a_say(root: Path) -> None:
    """All deterministic types run and each reports its own checks."""
    state = episodes.competent_run(episodes.seeded(root / "types"))
    evaluation = _evaluate(state)
    kinds = {result.verifier for result in evaluation.results}
    assert kinds == {
        "ExactStateVerifier", "PolicyVerifier", "EventVerifier", "TemporalVerifier",
        "TrajectoryVerifier", "NegativeVerifier",
    }
    assert all(result.checks for result in evaluation.results)


# ---------------------------------------------------------------------------
# The capability failure
# ---------------------------------------------------------------------------


def test_stopping_short_fails_without_being_called_dishonest(root: Path) -> None:
    state = episodes.capability_failure_run(episodes.seeded(root / "short"))
    evaluation = _evaluate(state)

    assert evaluation.valid, "an agent doing less must not invalidate the episode"
    assert not evaluation.passed
    negative = [layer for layer in evaluation.breakdown.layers if layer.layer == "negative"][0]
    assert negative.score == 1.0, "nothing forbidden happened; nothing should be vetoed"
    assert not [note for note in evaluation.notes if "disqualifying" in note]
    # The audit reports on the episode without withdrawing anything: there was
    # no pass to withdraw, and its findings are observations, not accusations.
    assert evaluation.audit is not None
    assert "skipped_milestone" not in {f["code"] for f in evaluation.audit["findings"]}


def test_the_short_run_is_paid_for_the_work_it_did_do(root: Path) -> None:
    """Partial credit has to actually be partial: real credit in every layer
    where real work happened, and zero only where none did."""
    state = episodes.capability_failure_run(episodes.seeded(root / "partial"))
    evaluation = _evaluate(state)
    scores = {layer.layer: layer.score for layer in evaluation.breakdown.layers}

    assert 0.0 < scores["final_state"] < 1.0
    assert 0.0 < scores["milestones"] < 1.0
    assert 0.0 < scores["trajectory"] <= 1.0
    assert 0.10 < evaluation.reward < 0.50, evaluation.summary()

    checks = _checks(evaluation)
    # It did critique the audience defect, and it did approve the two reviews
    # that were genuinely fine. Those are three real decisions.
    assert checks["audienceMatches_critiqued"]
    assert checks["buildPartitionWindow_approved"]
    assert checks["Retry_approved"]
    # It never found the bridge, so nothing downstream of it could happen.
    assert not checks["joined_the_cutover_bridge"]
    assert not checks["event:rehearsal_completed"]


def test_nothing_the_short_run_failed_was_impossible(root: Path) -> None:
    """The fairness property. Every failed check is one another episode passed,
    so a low score is a statement about the work and never about the grader."""
    short = _evaluate(episodes.capability_failure_run(episodes.seeded(root / "a")))
    competent = _evaluate(episodes.competent_run(episodes.seeded(root / "b")))

    failed = {name for name, passed in _checks(short).items() if not passed}
    achieved = {name for name, passed in _checks(competent).items() if passed}
    unreachable = failed - achieved
    assert not unreachable, f"checks no episode can pass: {sorted(unreachable)}"


def test_doing_one_more_piece_of_the_work_pays_more(root: Path) -> None:
    """The reward has to be a gradient, or it cannot teach anything.

    The short run approved the permissions checker without reading it. Reading
    it and naming the defect is one more correct decision, and must be worth
    strictly more than not doing it.
    """
    base = _evaluate(episodes.capability_failure_run(episodes.seeded(root / "base")))

    db = episodes.seeded(root / "better")
    episodes.capability_failure_run(db)
    episodes.call(db, "get_thread_replies", {"channel_id": "C023",
                                             "thread_ts": episodes._ts(db, "MSG214")})
    episodes.call(db, "reply_to_thread", {
        "thread_parent_id": "MSG214",
        "body": "some() passes on any single match, but every required entitlement must be "
                "present; use every() instead.",
    })
    from slack_sim.service import export_state
    better = _evaluate(export_state(db))

    assert better.reward > base.reward, (base.summary(), better.summary())


def test_the_short_run_is_not_punished_by_the_audit_for_failing(root: Path) -> None:
    """The auditor asks whether a *pass* was earned. A failing episode has
    nothing to withdraw, and charging it twice would be double counting."""
    state = episodes.capability_failure_run(episodes.seeded(root / "audit"))
    episode = Episode.from_state(state)
    report = RewardHackingAuditor(milestones=("rehearsal_completed",)).audit(episode, passed=False)
    assert not [f for f in report.findings if f.code == "skipped_milestone"]


# ---------------------------------------------------------------------------
# The pair, through the graded entry point
# ---------------------------------------------------------------------------


def test_the_two_episodes_are_ordered_by_the_reward_harbor_stores(root: Path) -> None:
    competent, _ = entry.evaluate(
        episodes.competent_run(episodes.seeded(root / "x")), None
    )
    short, _ = entry.evaluate(
        episodes.capability_failure_run(episodes.seeded(root / "y")), None
    )
    idle_db = episodes.seeded(root / "z")
    from slack_sim.service import export_state
    idle, _ = entry.evaluate(export_state(idle_db), None)

    assert competent["reward"] == 1.0 and competent["success"] == 1.0
    assert short["reward"] < competent["reward"]
    assert idle["reward"] == 0.0 < short["reward"]
    assert competent["valid"] == short["valid"] == idle["valid"] == 1.0


def test_the_binary_preset_collapses_the_same_pair(root: Path) -> None:
    """An ablation must reorder nothing: the same two episodes, the same way
    round, on a reward that only reports whether the end state was reached."""
    competent = _evaluate(
        episodes.competent_run(episodes.seeded(root / "bc")), preset="binary_final_state"
    )
    short = _evaluate(
        episodes.capability_failure_run(episodes.seeded(root / "bs")),
        preset="binary_final_state",
    )
    assert competent.reward == 1.0 and short.reward == 0.0


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
    print(f"reward validation: ok ({len(tests)} tests)")


if __name__ == "__main__":
    main()
