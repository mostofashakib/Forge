#!/usr/bin/env python3
"""The flat reward stream a harness stores, and why its key set never varies."""

from __future__ import annotations

import sys

from reward_episodes import CAPABILITY_FAILURE_RUN, COMPETENT_RUN, DISHONEST_RUN
from verifiers import Episode, TieredRewardEngine, blank_reward_dict, reward_dict, reward_keys
from verifiers.contracts.reassignment import build_contract
from verifiers.harbor import HEADLINE_KEYS, layer_keys
from verifiers.results import LAYERS
from workspace import ORACLE_STEPS, Workspace, run_steps

FAILURES: list[str] = []
CONTRACT = build_contract()


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def rewards_for(steps, preset: str | None = None) -> dict[str, float]:
    with Workspace() as workspace:
        run_steps(workspace, steps)
        evaluation = TieredRewardEngine.for_preset(CONTRACT, preset).evaluate(
            Episode.from_state(workspace.state())
        )
    flat = blank_reward_dict(1.0)
    flat.update(reward_dict(evaluation))
    return flat


def test_the_key_set_never_varies() -> None:
    """A missing key can only mean 'this run did not finish'."""
    expected = set(reward_keys())
    for name, steps in (
        ("oracle", ORACLE_STEPS), ("competent", COMPETENT_RUN),
        ("failure", CAPABILITY_FAILURE_RUN), ("dishonest", DISHONEST_RUN), ("nothing", ()),
    ):
        check(f"{name} declares every key", set(rewards_for(steps)) == expected,
              str(expected ^ set(rewards_for(steps))))
    check("a blank reward declares them too", set(blank_reward_dict(0.0)) == expected)


def test_the_keys_are_the_verdict_then_its_terms() -> None:
    check("the headline keys lead", set(HEADLINE_KEYS) <= set(reward_keys()))
    for layer in LAYERS:
        for key in (f"layer_{layer}", f"layer_weight_{layer}", f"layer_available_{layer}"):
            check(f"{key} is reported", key in reward_keys())
    check("every layer key is accounted for", set(layer_keys()) <= set(reward_keys()))


def test_every_value_is_a_number() -> None:
    """Harbor stores one number per key. A string here is a broken run report."""
    for name, steps in (("oracle", ORACLE_STEPS), ("dishonest", DISHONEST_RUN)):
        rewards = rewards_for(steps)
        bad = {k: v for k, v in rewards.items() if not isinstance(v, float)}
        check(f"{name}: all floats", not bad, str(bad))


def test_a_blank_reward_accuses_nobody() -> None:
    blank = blank_reward_dict(0.0)
    check("valid says why there is nothing here", blank["valid"] == 0.0)
    check("the reward is zero", blank["reward"] == 0.0)
    # Nothing was graded, so nothing was found suspicious. An audit failure for a
    # run that never happened would be an accusation.
    check("but the audit is not a failure", blank["audit_pass"] == 1.0)
    check("and no findings are claimed", blank["audit_findings"] == 0.0)


def test_the_headline_agrees_with_the_terms() -> None:
    for name, steps in (
        ("oracle", ORACLE_STEPS), ("failure", CAPABILITY_FAILURE_RUN),
        ("dishonest", DISHONEST_RUN),
    ):
        rewards = rewards_for(steps)
        derived = max(0.0, min(1.0, rewards["base_reward"] - rewards["penalty_total"]))
        check(f"{name}: base - penalties is the reward",
              abs(derived - rewards["reward"]) < 1e-6,
              f"{rewards['base_reward']} - {rewards['penalty_total']} != {rewards['reward']}")


def test_success_tracks_a_full_reward() -> None:
    check("the oracle succeeds", rewards_for(ORACLE_STEPS)["success"] == 1.0)
    check("a vetoed run does not", rewards_for(DISHONEST_RUN)["success"] == 0.0)
    check("nor does an incomplete one", rewards_for(CAPABILITY_FAILURE_RUN)["success"] == 0.0)


def test_a_preset_that_drops_a_layer_says_so() -> None:
    rewards = rewards_for(ORACLE_STEPS, "binary_final_state")
    check("the layer it does not run is reported unavailable",
          rewards["layer_available_milestones"] == 0.0, str(rewards["layer_available_milestones"]))
    check("rather than as a zero score it never earned",
          rewards["layer_weight_milestones"] == 0.0)
    check("and the run still reports a reward", rewards["reward"] == 1.0)


def main() -> int:
    print(__doc__)
    for test in (
        test_the_key_set_never_varies,
        test_the_keys_are_the_verdict_then_its_terms,
        test_every_value_is_a_number,
        test_a_blank_reward_accuses_nobody,
        test_the_headline_agrees_with_the_terms,
        test_success_tracks_a_full_reward,
        test_a_preset_that_drops_a_layer_says_so,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all Harbor-reward checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
