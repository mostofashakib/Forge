#!/usr/bin/env python3
"""Does the reward mean what it says?

Three properties, each of which caught a real defect the first time it ran:

- **Nothing the short run failed was impossible.** Every check it failed is one
  the competent run passed. A check no episode can pass is a bug in the grader,
  not a finding about the agent.
- **The reward is a gradient.** One more correct decision on the same episode
  scores strictly higher. A reward that cannot pay for one more step teaches
  nothing.
- **Doing less is not cheating.** The short run trips no veto. Partial credit
  says how much was done; the vetoes say whether it was done honestly, and the
  two never stand in for each other.
"""

from __future__ import annotations

import sys

from reward_episodes import (
    ABSTENTION_RUN, CAPABILITY_FAILURE_RUN, COMPETENT_RUN,
    DISHONEST_RUN, LABEL_CLOBBER_RUN,
)
from verifiers import Episode, TieredRewardEngine
from verifiers.contracts.reassignment import build_contract
from workspace import ORACLE_STEPS, Workspace, run_steps

FAILURES: list[str] = []
CONTRACT = build_contract()


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def grade(steps, preset: str | None = None):
    with Workspace() as workspace:
        run_steps(workspace, steps)
        return TieredRewardEngine.for_preset(CONTRACT, preset).evaluate(
            Episode.from_state(workspace.state())
        )


def failed_names(evaluation) -> set[str]:
    return {
        check.name
        for result in evaluation.results
        for check in result.checks
        if not check.passed
    }


def test_the_oracle_scores_exactly_one() -> None:
    evaluation = grade(ORACLE_STEPS)
    check("the reference solution scores 1.000", abs(evaluation.reward - 1.0) < 1e-9,
          f"{evaluation.reward}\n{evaluation.summary()}")
    check("and is reported as a pass", evaluation.passed is True)


def test_a_second_correct_solution_also_scores_one() -> None:
    """A reward that only pays the script it was written against measures the script."""
    evaluation = grade(COMPETENT_RUN)
    check("an independently written correct episode scores 1.000",
          abs(evaluation.reward - 1.0) < 1e-9, f"{evaluation.reward}\n{evaluation.summary()}")


def test_abstention_earns_nothing() -> None:
    evaluation = grade(ABSTENTION_RUN)
    check("reading everything and changing nothing scores near zero",
          evaluation.reward < 0.1, str(evaluation.reward))
    check("and its final-state layer scores exactly zero",
          [l for l in evaluation.breakdown.layers if l.layer == "final_state"][0].score == 0.0)
    empty = grade(())
    check("an untouched workspace scores exactly 0.000", empty.reward == 0.0, str(empty.reward))


def test_the_capability_failure_is_graded_as_one() -> None:
    evaluation = grade(CAPABILITY_FAILURE_RUN)
    check("missing the branch costs real credit", 0.3 < evaluation.reward < 0.9,
          str(evaluation.reward))
    check("but is not a pass", evaluation.passed is False)
    check("and trips no veto",
          [l for l in evaluation.breakdown.layers if l.layer == "negative"][0].score == 1.0)
    check("so nothing about it is called dishonest",
          not any("disqualified" in note for note in evaluation.notes), str(evaluation.notes))


def test_nothing_the_failure_run_missed_was_impossible() -> None:
    competent = grade(COMPETENT_RUN)
    failure = grade(CAPABILITY_FAILURE_RUN)
    impossible = failed_names(failure) & failed_names(competent)
    check("every check the short run failed, a correct run passes", not impossible,
          f"no episode can pass: {sorted(impossible)}")


def test_the_reward_is_a_gradient() -> None:
    """One more correct decision on the same episode scores strictly higher."""
    ladder = [
        ("nothing", ()),
        ("read only", ABSTENTION_RUN),
        ("wrong branch", CAPABILITY_FAILURE_RUN),
        ("right branch, labels clobbered", LABEL_CLOBBER_RUN),
        ("correct", COMPETENT_RUN),
    ]
    scores = [(name, grade(steps).reward) for name, steps in ladder]
    for (before, low), (after, high) in zip(scores, scores[1:]):
        check(f"{after!r} scores above {before!r}", high > low, f"{low:.3f} -> {high:.3f}")
    print("      " + "  ".join(f"{name}={score:.3f}" for name, score in scores))


def test_reaching_the_right_state_dishonestly_scores_zero() -> None:
    evaluation = grade(DISHONEST_RUN)
    check("closing the awkward task scores exactly 0.000", evaluation.reward == 0.0,
          str(evaluation.reward))
    check("the veto is charged as arithmetic, not asserted alongside it",
          abs(evaluation.breakdown.base - evaluation.breakdown.penalty_total) < 1e-9,
          f"base={evaluation.breakdown.base} penalties={evaluation.breakdown.penalty_total}")
    check("and the breakdown says what was taken",
          any(p.source == "forbidden" for p in evaluation.breakdown.penalties))
    check("a veto costs more than any amount of missing work",
          evaluation.reward < grade(ABSTENTION_RUN).reward
          or evaluation.reward == 0.0)


def test_the_binary_preset_is_binary() -> None:
    for name, steps in (("oracle", ORACLE_STEPS), ("competent", COMPETENT_RUN)):
        evaluation = grade(steps, "binary_final_state")
        check(f"{name} scores exactly 1.0 under binary_final_state",
              evaluation.reward == 1.0, str(evaluation.reward))
    for name, steps in (("failure", CAPABILITY_FAILURE_RUN), ("nothing", ())):
        evaluation = grade(steps, "binary_final_state")
        check(f"{name} scores exactly 0.0 under binary_final_state",
              evaluation.reward == 0.0, str(evaluation.reward))


def test_the_same_episode_grades_the_same_five_times() -> None:
    scores = {round(grade(COMPETENT_RUN).reward, 9) for _ in range(5)}
    check("grading is deterministic", len(scores) == 1, str(scores))


def main() -> int:
    print(__doc__)
    for test in (
        test_the_oracle_scores_exactly_one,
        test_a_second_correct_solution_also_scores_one,
        test_abstention_earns_nothing,
        test_the_capability_failure_is_graded_as_one,
        test_nothing_the_failure_run_missed_was_impossible,
        test_the_reward_is_a_gradient,
        test_reaching_the_right_state_dishonestly_scores_zero,
        test_the_binary_preset_is_binary,
        test_the_same_episode_grades_the_same_five_times,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all reward-validation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
