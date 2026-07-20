"""Reward → advantage (GRPO) and reward → label (DPO) mapping.

The mapping must be a deterministic function of the graded rewards that were
already assigned to each trajectory, and it must produce NO training signal when
there is nothing to learn (a group of equally-rewarded rollouts, or a preference
pair with a tie).
"""
from __future__ import annotations

import math

from forge.training.dataset import PreferenceRecord, RolloutRecord
from forge.training.reward_mapping import dpo_examples, grpo_advantages


def _rollout(task: str, reward: float, *, passed: bool = True, completion: str = "c") -> RolloutRecord:
    return RolloutRecord(
        episode_id=f"ep-{task}-{reward}",
        task_name=task,
        prompt=f"Task: {task}",
        completion=completion,
        total_reward=reward,
        passed=passed,
        per_step_rewards=[reward],
    )


# ---------------------------------------------------------------------------
# GRPO: group-relative advantage
# ---------------------------------------------------------------------------

def test_grpo_advantage_is_group_relative_and_zero_mean():
    rollouts = [_rollout("t", 0.0), _rollout("t", 1.0), _rollout("t", 2.0)]
    examples = grpo_advantages(rollouts)
    assert len(examples) == 3
    # Advantages are the rewards standardized within the prompt group.
    advs = sorted(e.advantage for e in examples)
    assert math.isclose(sum(advs), 0.0, abs_tol=1e-9)
    assert advs[0] < 0 < advs[-1]  # worst below mean, best above


def test_grpo_is_deterministic():
    rollouts = [_rollout("t", 0.0), _rollout("t", 1.0), _rollout("t", 3.0)]
    a = [(e.completion, round(e.advantage, 6)) for e in grpo_advantages(rollouts)]
    b = [(e.completion, round(e.advantage, 6)) for e in grpo_advantages(rollouts)]
    assert a == b


def test_grpo_groups_by_prompt():
    rollouts = [
        _rollout("a", 0.0), _rollout("a", 1.0),
        _rollout("b", 5.0), _rollout("b", 7.0),
    ]
    examples = grpo_advantages(rollouts)
    by_prompt: dict[str, list[float]] = {}
    for e in examples:
        by_prompt.setdefault(e.prompt, []).append(e.advantage)
    # Each prompt group is standardized independently → each sums to ~0.
    for advs in by_prompt.values():
        assert math.isclose(sum(advs), 0.0, abs_tol=1e-9)


def test_grpo_equal_reward_group_yields_no_examples():
    # False-positive guard: a group whose rollouts all scored the same has no
    # relative signal and must NOT produce a training example.
    rollouts = [_rollout("t", 1.0), _rollout("t", 1.0), _rollout("t", 1.0)]
    assert grpo_advantages(rollouts) == []


def test_grpo_singleton_group_yields_no_examples():
    # One rollout for a prompt has no group to be relative to.
    assert grpo_advantages([_rollout("solo", 1.0)]) == []


def test_grpo_empty_input_yields_no_examples():
    assert grpo_advantages([]) == []


# ---------------------------------------------------------------------------
# DPO: preference labels
# ---------------------------------------------------------------------------

def _pref(chosen_r: float, rejected_r: float, *, task: str = "t") -> PreferenceRecord:
    return PreferenceRecord(
        task=task,
        prompt=f"Task: {task}",
        chosen="good",
        rejected="bad",
        chosen_reward=chosen_r,
        rejected_reward=rejected_r,
        chosen_passed=chosen_r > 0,
        rejected_passed=False,
    )


def test_dpo_emits_chosen_over_rejected():
    examples = dpo_examples([_pref(2.0, 0.0)])
    assert len(examples) == 1
    ex = examples[0]
    assert ex.chosen == "good" and ex.rejected == "bad"
    assert ex.prompt == "Task: t"


def test_dpo_skips_tied_pairs():
    # False-positive guard: chosen and rejected scored identically → no
    # preference signal → must NOT produce an example.
    assert dpo_examples([_pref(1.0, 1.0)]) == []


def test_dpo_skips_inverted_pairs():
    # Defensive: a malformed pair where "chosen" scored lower is dropped.
    assert dpo_examples([_pref(0.0, 1.0)]) == []


def test_dpo_empty_input_yields_no_examples():
    assert dpo_examples([]) == []
