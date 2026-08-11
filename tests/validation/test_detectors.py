"""Statistical detectors for trajectory pathologies.

Distribution drift, reward collapse, and outlier episodes are statistics, not
questions for a language model: the statistical answer is reproducible, free,
instant, and correct, and the LLM's is none of those.
"""
from __future__ import annotations

import pytest

from forge.validation.detectors import (
    EpisodeFeatures,
    analyze_episodes,
    detect_anomalous_episodes,
    detect_distribution_drift,
    detect_reward_collapse,
    detect_reward_hacking,
    population_stability_index,
)


def _episodes(rewards, steps=None, actions=None) -> list[EpisodeFeatures]:
    steps = steps or [10] * len(rewards)
    actions = actions or [["a", "b", "c"]] * len(rewards)
    return [
        EpisodeFeatures(episode_id=f"ep{i}", reward=r, steps=s, actions=list(acts))
        for i, (r, s, acts) in enumerate(zip(rewards, steps, actions))
    ]


# ---------------------------------------------------------------------------
# Population stability index
# ---------------------------------------------------------------------------

def test_identical_distributions_have_no_instability():
    psi = population_stability_index({"a": 5, "b": 5}, {"a": 5, "b": 5})

    assert psi == pytest.approx(0.0, abs=1e-9)


def test_a_shifted_distribution_registers_instability():
    psi = population_stability_index({"a": 9, "b": 1}, {"a": 1, "b": 9})

    assert psi > 0.25


def test_a_category_absent_from_one_side_does_not_divide_by_zero():
    psi = population_stability_index({"a": 10}, {"b": 10})

    assert psi > 0.0
    assert psi == psi  # not NaN


def test_an_empty_distribution_is_perfectly_stable_rather_than_undefined():
    assert population_stability_index({}, {}) == 0.0


# ---------------------------------------------------------------------------
# Reward collapse
# ---------------------------------------------------------------------------

def test_a_sustained_reward_drop_is_detected():
    finding = detect_reward_collapse(_episodes([0.9, 0.9, 0.85, 0.1, 0.05, 0.1]))

    assert finding is not None
    assert finding.category == "reward_collapse"


def test_stable_rewards_produce_no_collapse_finding():
    assert detect_reward_collapse(_episodes([0.7, 0.72, 0.68, 0.71, 0.69, 0.70])) is None


def test_rising_rewards_are_not_reported_as_collapse():
    """False-positive guard: improvement is a change, not a collapse."""
    assert detect_reward_collapse(_episodes([0.1, 0.2, 0.3, 0.8, 0.85, 0.9])) is None


def test_too_few_episodes_cannot_establish_a_collapse():
    """False-positive guard: two points are a line, not a trend."""
    assert detect_reward_collapse(_episodes([0.9, 0.1])) is None


def test_noisy_but_trendless_rewards_are_not_a_collapse():
    assert detect_reward_collapse(_episodes([0.9, 0.1, 0.9, 0.1, 0.9, 0.1])) is None


# ---------------------------------------------------------------------------
# Distribution drift
# ---------------------------------------------------------------------------

def test_a_changed_action_vocabulary_is_detected_as_drift():
    early = [["ls", "cat", "ls"]] * 3
    late = [["curl", "wget", "curl"]] * 3
    finding = detect_distribution_drift(_episodes([0.5] * 6, actions=early + late))

    assert finding is not None
    assert finding.category == "distribution_drift"


def test_a_stable_action_vocabulary_produces_no_drift_finding():
    actions = [["ls", "cat", "grep"]] * 6

    assert detect_distribution_drift(_episodes([0.5] * 6, actions=actions)) is None


def test_too_few_episodes_cannot_establish_drift():
    actions = [["ls"], ["curl"]]

    assert detect_distribution_drift(_episodes([0.5, 0.5], actions=actions)) is None


def test_episodes_with_no_actions_do_not_register_drift():
    assert detect_distribution_drift(_episodes([0.5] * 6, actions=[[]] * 6)) is None


# ---------------------------------------------------------------------------
# Anomalous episodes
# ---------------------------------------------------------------------------

def test_an_outlier_episode_is_flagged():
    episodes = _episodes([0.5] * 9 + [0.5], steps=[10] * 9 + [900])
    findings = detect_anomalous_episodes(episodes)

    assert any(f.category == "anomalous_pattern" for f in findings)
    assert "ep9" in findings[0].episode_ids


def test_uniform_episodes_produce_no_anomalies():
    assert detect_anomalous_episodes(_episodes([0.5] * 10)) == []


def test_a_small_sample_does_not_manufacture_outliers():
    """False-positive guard: with three points, everything looks extreme."""
    assert detect_anomalous_episodes(_episodes([0.1, 0.5, 0.9])) == []


def test_a_wide_but_evenly_spread_sample_is_not_anomalous():
    """False-positive guard: high variance is not the same as an outlier."""
    rewards = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    assert detect_anomalous_episodes(_episodes(rewards)) == []


# ---------------------------------------------------------------------------
# Reward hacking
# ---------------------------------------------------------------------------

def test_high_reward_from_a_trivial_trajectory_is_flagged():
    episodes = _episodes(
        [0.4, 0.45, 0.5, 0.4, 0.45, 1.0],
        steps=[40, 45, 50, 40, 45, 1],
    )
    findings = detect_reward_hacking(episodes)

    assert any(f.category == "reward_hacking" for f in findings)
    assert "ep5" in findings[0].episode_ids


def test_high_reward_from_a_normal_length_trajectory_is_not_flagged():
    """False-positive guard: succeeding well is not the same as cheating."""
    episodes = _episodes([0.4, 0.45, 0.5, 0.4, 0.45, 1.0], steps=[40, 45, 50, 40, 45, 44])

    assert detect_reward_hacking(episodes) == []


def test_a_short_episode_with_low_reward_is_not_flagged():
    episodes = _episodes([0.4, 0.45, 0.5, 0.4, 0.45, 0.05], steps=[40, 45, 50, 40, 45, 1])

    assert detect_reward_hacking(episodes) == []


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def test_clean_episodes_yield_no_findings():
    assert analyze_episodes(_episodes([0.7] * 8)) == []


def test_analysis_is_deterministic_across_repeated_calls():
    """The whole reason to prefer statistics: the same input gives the same answer."""
    episodes = _episodes([0.9, 0.9, 0.85, 0.1, 0.05, 0.1])

    first = [f.description for f in analyze_episodes(episodes)]
    second = [f.description for f in analyze_episodes(episodes)]

    assert first == second
    assert first != []


def test_analysis_of_an_empty_run_is_empty_not_an_error():
    assert analyze_episodes([]) == []
