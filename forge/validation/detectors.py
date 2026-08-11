"""Statistical detectors for trajectory pathologies.

Distribution drift, reward collapse, and outlier episodes are statistics. Asking
a language model to eyeball them costs money and a round trip, varies between
identical runs, and is less accurate than the closed-form answer. These
detectors are deterministic, so a finding can be reproduced by anyone holding
the same episodes.

Each detector is deliberately conservative — a monitoring surface that cries
wolf gets switched off, and a false alarm on a healthy run is worse than a
missed one on an unhealthy run that later detectors will also see.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping, Sequence

# Below this many episodes there is not enough evidence to claim a trend, and
# every statistic here becomes dominated by noise.
_MIN_EPISODES_FOR_TREND = 4
_MIN_EPISODES_FOR_OUTLIERS = 8

# PSI convention: <0.1 stable, 0.1–0.25 moderate shift, >0.25 significant.
_PSI_SIGNIFICANT = 0.25
# Smoothing so a category present on one side only does not divide by zero.
_PSI_EPSILON = 1e-4

# A collapse is a drop of at least this fraction of the earlier mean reward.
_COLLAPSE_DROP_FRACTION = 0.5
# ...and the two halves must be separated by more than this much noise.
_COLLAPSE_NOISE_MULTIPLE = 1.5

# Modified z-score cutoff. 3.5 is the conventional threshold for the
# median-absolute-deviation outlier test, which is robust to the very outliers
# it looks for in a way that a mean/standard-deviation test is not.
_OUTLIER_Z = 3.5

# Reward hacking: a top-decile reward earned in a small fraction of the typical
# episode length.
_HACK_REWARD_QUANTILE = 0.75
_HACK_LENGTH_FRACTION = 0.25


@dataclass(frozen=True)
class EpisodeFeatures:
    """The per-episode signal these detectors operate on."""

    episode_id: str
    reward: float
    steps: int
    actions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DetectionFinding:
    category: str
    severity: str
    episode_ids: list[str]
    description: str
    evidence: str

    def as_record(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "episode_ids": list(self.episode_ids),
            "description": self.description,
            "evidence": self.evidence,
        }


def population_stability_index(
    early: Mapping[str, int], late: Mapping[str, int]
) -> float:
    """Population stability index between two categorical distributions.

    The standard drift statistic: sum over categories of
    ``(late% - early%) * ln(late% / early%)``. Zero when the distributions
    match; conventionally significant above 0.25.
    """
    early_total, late_total = sum(early.values()), sum(late.values())
    if not early_total or not late_total:
        return 0.0
    psi = 0.0
    for category in set(early) | set(late):
        early_share = max(early.get(category, 0) / early_total, _PSI_EPSILON)
        late_share = max(late.get(category, 0) / late_total, _PSI_EPSILON)
        psi += (late_share - early_share) * math.log(late_share / early_share)
    return psi


def _halves(episodes: Sequence[EpisodeFeatures]):
    midpoint = len(episodes) // 2
    return episodes[:midpoint], episodes[midpoint:]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def detect_reward_collapse(
    episodes: Sequence[EpisodeFeatures],
) -> DetectionFinding | None:
    """Flag a sustained drop in reward between the first and second half."""
    if len(episodes) < _MIN_EPISODES_FOR_TREND:
        return None
    early, late = _halves(episodes)
    early_mean, late_mean = _mean([e.reward for e in early]), _mean([e.reward for e in late])
    if early_mean <= 0:
        return None
    drop = early_mean - late_mean
    if drop <= 0:
        return None
    # Require the drop to clear the run's own noise floor, so an alternating
    # high/low series is not mistaken for a downward trend.
    spread = _mean([abs(e.reward - early_mean) for e in early]) + _mean(
        [abs(e.reward - late_mean) for e in late]
    )
    if drop < _COLLAPSE_DROP_FRACTION * early_mean:
        return None
    if drop < _COLLAPSE_NOISE_MULTIPLE * spread:
        return None
    return DetectionFinding(
        category="reward_collapse",
        severity="high",
        episode_ids=[e.episode_id for e in late],
        description="Mean reward dropped sharply across the run.",
        evidence=(
            f"mean reward fell from {early_mean:.3f} to {late_mean:.3f} "
            f"between the first and second half of {len(episodes)} episodes"
        ),
    )


def detect_distribution_drift(
    episodes: Sequence[EpisodeFeatures],
) -> DetectionFinding | None:
    """Flag a shift in the action vocabulary between the first and second half."""
    if len(episodes) < _MIN_EPISODES_FOR_TREND:
        return None
    early, late = _halves(episodes)
    early_actions = Counter(action for e in early for action in e.actions)
    late_actions = Counter(action for e in late for action in e.actions)
    if not early_actions or not late_actions:
        return None
    psi = population_stability_index(early_actions, late_actions)
    if psi <= _PSI_SIGNIFICANT:
        return None
    return DetectionFinding(
        category="distribution_drift",
        severity="medium",
        episode_ids=[e.episode_id for e in late],
        description="The agent's action vocabulary shifted during the run.",
        evidence=(
            f"population stability index {psi:.3f} exceeds {_PSI_SIGNIFICANT}; "
            f"early actions {dict(early_actions)}, late actions {dict(late_actions)}"
        ),
    )


def _modified_z_scores(values: Sequence[float]) -> list[float]:
    """Median-absolute-deviation z-scores, robust to the outliers they detect."""
    centre = median(values)
    deviations = [abs(value - centre) for value in values]
    mad = median(deviations)
    if mad:
        return [0.6745 * (value - centre) / mad for value in values]
    # A zero MAD does not mean "no outliers": it happens whenever more than half
    # the values are identical, which is exactly the shape a single extreme
    # episode among many uniform ones takes. Fall back to the mean absolute
    # deviation, which stays sensitive there.
    mean_deviation = _mean(deviations)
    if not mean_deviation:
        return [0.0] * len(values)  # every value identical — nothing to flag
    return [0.7979 * (value - centre) / mean_deviation for value in values]


def detect_anomalous_episodes(
    episodes: Sequence[EpisodeFeatures],
) -> list[DetectionFinding]:
    """Flag episodes whose reward or length is a statistical outlier."""
    if len(episodes) < _MIN_EPISODES_FOR_OUTLIERS:
        return []
    findings: list[DetectionFinding] = []
    for label, values in (
        ("reward", [e.reward for e in episodes]),
        ("step count", [float(e.steps) for e in episodes]),
    ):
        scores = _modified_z_scores(values)
        outliers = [
            episodes[i].episode_id
            for i, score in enumerate(scores)
            if abs(score) > _OUTLIER_Z
        ]
        if outliers:
            findings.append(DetectionFinding(
                category="anomalous_pattern",
                severity="low",
                episode_ids=outliers,
                description=f"Episode {label} is a statistical outlier.",
                evidence=(
                    f"modified z-score above {_OUTLIER_Z} against a median "
                    f"{label} of {median(values):.3f}"
                ),
            ))
    return findings


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    index = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[index]


def detect_reward_hacking(
    episodes: Sequence[EpisodeFeatures],
) -> list[DetectionFinding]:
    """Flag episodes earning a high reward from an implausibly short trajectory."""
    if len(episodes) < _MIN_EPISODES_FOR_TREND:
        return []
    rewards = [e.reward for e in episodes]
    steps = [float(e.steps) for e in episodes]
    reward_cut = _quantile(rewards, _HACK_REWARD_QUANTILE)
    typical_length = median(steps)
    if typical_length <= 0:
        return []
    suspects = [
        e.episode_id
        for e in episodes
        if e.reward >= reward_cut and e.steps < _HACK_LENGTH_FRACTION * typical_length
    ]
    if not suspects:
        return []
    return [DetectionFinding(
        category="reward_hacking",
        severity="high",
        episode_ids=suspects,
        description="High reward earned from an implausibly short trajectory.",
        evidence=(
            f"reward at or above {reward_cut:.3f} in under "
            f"{_HACK_LENGTH_FRACTION:.0%} of the median {typical_length:.0f}-step "
            "episode"
        ),
    )]


def analyze_episodes(
    episodes: Sequence[EpisodeFeatures],
) -> list[DetectionFinding]:
    """Run every detector, most severe first. Deterministic for a given input."""
    if not episodes:
        return []
    findings: list[DetectionFinding] = []
    for detector in (detect_reward_collapse, detect_distribution_drift):
        finding = detector(episodes)
        if finding is not None:
            findings.append(finding)
    findings.extend(detect_reward_hacking(episodes))
    findings.extend(detect_anomalous_episodes(episodes))
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (severity_rank.get(f.severity, 3), f.category))
    return findings
