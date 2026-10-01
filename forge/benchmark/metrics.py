"""Benchmark evaluation metrics and graph generation (Feature Request 9).

Includes:
- pass@1, pass@k (unbiased combinatorial estimator)
- pass^k (all-k pass / robustness estimator)
- Cohen's kappa agreement rate
- Diagnostic analytics to detect reward hacking, distinguish genuine learning from
  memorization, and identify data contamination.
"""
from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the unbiased estimator of pass@k (HumanEval / Chen et al.).

    Args:
        n: Total number of samples generated per problem.
        c: Number of correct (passing) samples.
        k: Sample threshold (k <= n).

    Raises:
        ValueError: when n or k is not positive, or c is outside [0, n].
    """
    if n <= 0 or k <= 0:
        raise ValueError(f"pass@k needs n >= 1 and k >= 1, got n={n}, k={k}")
    if not 0 <= c <= n:
        raise ValueError(f"pass@k needs 0 <= c <= n, got c={c}, n={n}")
    if n < k:
        return float(c >= 1)
    if c >= n:
        return 1.0
    if n - c < k:
        return 1.0
    comb_total = math.comb(n, k)
    if comb_total == 0:
        return 0.0
    comb_fail = math.comb(n - c, k)
    return 1.0 - (comb_fail / comb_total)


def compute_pass_pow_k(n: int, c: int, k: int) -> float:
    """Compute pass^k (probability that all k drawn samples pass).

    Measures consistency and resilience against reward hacking / lucky guessing.
    """
    if n <= 0 or k <= 0:
        return 0.0
    if n < k or c < k:
        return 0.0
    comb_total = math.comb(n, k)
    if comb_total == 0:
        return 0.0
    comb_pass = math.comb(c, k)
    return comb_pass / comb_total


def compute_cohens_kappa(
    eval1: list[bool | int], eval2: list[bool | int]
) -> float:
    """Compute Cohen's kappa agreement rate between two evaluators/runs on the same task items.

    kappa = (p_o - p_e) / (1 - p_e)
    """
    if not eval1 or len(eval1) != len(eval2):
        return 0.0

    total = len(eval1)
    b1 = [bool(x) for x in eval1]
    b2 = [bool(x) for x in eval2]

    # Contingency counts
    yy = sum(1 for x, y in zip(b1, b2) if x and y)
    yn = sum(1 for x, y in zip(b1, b2) if x and not y)
    ny = sum(1 for x, y in zip(b1, b2) if not x and y)
    nn = sum(1 for x, y in zip(b1, b2) if not x and not y)

    p_o = (yy + nn) / total

    # Marginal probabilities
    p1_yes = (yy + yn) / total
    p1_no = 1.0 - p1_yes
    p2_yes = (yy + ny) / total
    p2_no = 1.0 - p2_yes

    p_e = (p1_yes * p2_yes) + (p1_no * p2_no)

    if math.isclose(1.0, p_e):
        return 1.0 if math.isclose(p_o, 1.0) else 0.0

    return (p_o - p_e) / (1.0 - p_e)


class TaskTrial(BaseModel):
    task_id: str
    passed_samples: int = Field(..., ge=0)
    total_samples: int = Field(..., ge=1)
    # Optional dual evaluation / paraphrase comparison
    paraphrased_passed_samples: int | None = None
    paraphrased_total_samples: int | None = None
    # Verifier agreement tracking
    verifier_passes: list[bool] = Field(default_factory=list)
    ground_truth_passes: list[bool] = Field(default_factory=list)
    ngram_overlap: float | None = None


class GraphSeries(BaseModel):
    name: str
    data: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkGraphData(BaseModel):
    k_labels: list[int]
    pass_at_k: list[float]
    pass_pow_k: list[float]
    cohens_kappa: float
    gap_pass_vs_pow: list[float]
    diagnostics: dict[str, Any]
    chart_configs: dict[str, Any]


def generate_benchmark_graphs(
    trials: list[TaskTrial],
    max_k: int | None = None,
) -> BenchmarkGraphData:
    """Generate comprehensive benchmark graph options and diagnostic signals."""
    if not trials:
        # No samples means no measurement, not a low risk.
        raise ValueError("no trials to chart")

    # Determine feasible range of k
    min_total = min(t.total_samples for t in trials)
    limit_k = max_k if max_k is not None else min(min_total, 10)
    limit_k = max(1, limit_k)
    k_range = list(range(1, limit_k + 1))

    pass_at_k_series, pass_pow_k_series, gap_series = _pass_curves(trials, k_range)
    kappa, has_verifier_pairs, orig_binary, para_binary = _agreement(trials)

    # Reward hacking: a large gap between pass@k and pass^k, or verifiers that
    # disagree with ground truth.
    pass1 = pass_at_k_series[0] if pass_at_k_series else 0.0
    pass_pow_max = pass_pow_k_series[-1] if pass_pow_k_series else 0.0
    gap_max = (pass_at_k_series[-1] if pass_at_k_series else 0.0) - pass_pow_max
    if gap_max > 0.6 or (has_verifier_pairs and kappa < 0.3):
        reward_hacking_risk = "high"
    elif gap_max > 0.35 or (has_verifier_pairs and kappa < 0.6):
        reward_hacking_risk = "moderate"
    else:
        reward_hacking_risk = "low"

    # Memorization: performance drops from original to paraphrased items.
    mem_gap = 0.0
    if orig_binary and para_binary:
        mem_gap = max(0.0, sum(orig_binary) / len(orig_binary) - sum(para_binary) / len(para_binary))
    memorization_risk = _tier(mem_gap, high=0.3, moderate=0.15)
    contamination_risk = _contamination_risk(trials)

    # Genuine learning index: balances high pass@1, robustness (pass^k), and generalization (1 - mem_gap)
    genuine_learning_index = round(
        max(0.0, min(1.0, (pass1 * 0.4 + pass_pow_max * 0.3 + (1.0 - mem_gap) * 0.3))),
        4,
    )
    diagnostics = {
        "reward_hacking_risk": reward_hacking_risk,
        "memorization_risk": memorization_risk,
        "contamination_risk": contamination_risk,
        "cohens_agreement_rate": round(kappa, 4),
        "memorization_performance_gap": round(mem_gap, 4),
        "robustness_gap_k": round(gap_max, 4),
        "genuine_learning_index": genuine_learning_index,
    }
    chart_configs = _chart_configs(
        k_range, pass_at_k_series, pass_pow_k_series, gap_series,
        kappa=kappa, genuine_learning_index=genuine_learning_index,
        risks=(reward_hacking_risk, memorization_risk, contamination_risk), gap_max=gap_max,
    )

    return BenchmarkGraphData(
        k_labels=k_range,
        pass_at_k=pass_at_k_series,
        pass_pow_k=pass_pow_k_series,
        cohens_kappa=round(kappa, 4),
        gap_pass_vs_pow=gap_series,
        diagnostics=diagnostics,
        chart_configs=chart_configs,
    )


def _pass_curves(trials: list[TaskTrial], k_range: list[int]) -> tuple[list[float], list[float], list[float]]:
    """Mean pass@k, pass^k and their gap across tasks, for each k."""
    pass_at_k: list[float] = []
    pass_pow_k: list[float] = []
    gaps: list[float] = []
    for k in k_range:
        mean_at_k = sum(compute_pass_at_k(t.total_samples, t.passed_samples, k) for t in trials) / len(trials)
        mean_pow_k = sum(compute_pass_pow_k(t.total_samples, t.passed_samples, k) for t in trials) / len(trials)
        pass_at_k.append(round(mean_at_k, 4))
        pass_pow_k.append(round(mean_pow_k, 4))
        gaps.append(round(mean_at_k - mean_pow_k, 4))
    return pass_at_k, pass_pow_k, gaps


def _agreement(trials: list[TaskTrial]) -> tuple[float, bool, list[bool], list[bool]]:
    """Cohen's kappa, preferring verifier vs ground truth, then original vs paraphrased.

    Returns (kappa, whether verifier/ground-truth pairs exist, original outcomes,
    paraphrased outcomes).
    """
    verifier: list[bool] = []
    ground_truth: list[bool] = []
    for t in trials:
        if t.verifier_passes and t.ground_truth_passes:
            verifier.extend(t.verifier_passes)
            ground_truth.extend(t.ground_truth_passes)
    paraphrased = [
        t for t in trials
        if t.paraphrased_total_samples is not None and t.paraphrased_passed_samples is not None
    ]
    original_binary = [t.passed_samples > 0 for t in paraphrased]
    paraphrased_binary = [t.paraphrased_passed_samples > 0 for t in paraphrased]

    if verifier and len(verifier) == len(ground_truth):
        kappa = compute_cohens_kappa(verifier, ground_truth)
    elif original_binary:
        kappa = compute_cohens_kappa(original_binary, paraphrased_binary)
    else:
        # Measure inter-seed/trial consistency across tasks
        kappa = compute_cohens_kappa(
            [t.passed_samples >= (t.total_samples / 2) for t in trials],
            [t.passed_samples > 0 for t in trials],
        )
    return kappa, bool(verifier), original_binary, paraphrased_binary


def _tier(value: float, *, high: float, moderate: float) -> str:
    if value > high:
        return "high"
    if value > moderate:
        return "moderate"
    return "low"


def _contamination_risk(trials: list[TaskTrial]) -> str:
    """Tasks with high n-gram overlap passing more often suggests contamination."""
    high = [t.passed_samples / t.total_samples for t in trials if t.ngram_overlap is not None and t.ngram_overlap > 0.5]
    low = [t.passed_samples / t.total_samples for t in trials if t.ngram_overlap is not None and t.ngram_overlap <= 0.5]
    if not (high and low):
        return "low"
    return _tier(sum(high) / len(high) - sum(low) / len(low), high=0.35, moderate=0.15)


_SEVERITY = {"high": 1.0, "moderate": 0.5, "low": 0.1}


def _chart_configs(
    k_range: list[int],
    pass_at_k: list[float],
    pass_pow_k: list[float],
    gaps: list[float],
    *,
    kappa: float,
    genuine_learning_index: float,
    risks: tuple[str, str, str],
    gap_max: float,
) -> dict:
    """Chart configurations formatted for frontend graphs (Chart.js / JSON schema)."""
    return {
        "pass_curve": {
            "type": "line",
            "title": "Pass@k vs Pass^k Robustness Curve",
            "xAxis": {"title": "k (Samples Drawn)", "categories": [f"k={k}" for k in k_range]},
            "yAxis": {"title": "Probability", "min": 0.0, "max": 1.0},
            "series": [
                {"name": "pass@k (At least 1 pass)", "data": pass_at_k, "color": "#10b981"},
                {"name": "pass^k (All k pass)", "data": pass_pow_k, "color": "#6366f1"},
                {"name": "Robustness Gap (pass@k - pass^k)", "data": gaps, "color": "#f59e0b"},
            ],
        },
        "agreement_bar": {
            "type": "bar",
            "title": "Agreement & Consistency (Cohen's Kappa)",
            "categories": ["Inter-Evaluator Kappa", "Genuine Learning Index"],
            "series": [
                {
                    "name": "Score",
                    "data": [round(kappa, 4), genuine_learning_index],
                    "colors": ["#3b82f6", "#10b981"],
                }
            ],
        },
        "risk_breakdown": {
            "type": "radar",
            "title": "Risk Diagnostics",
            "categories": ["Reward Hacking", "Memorization", "Contamination", "Robustness Gap"],
            "series": [
                {
                    "name": "Risk Severity",
                    "data": [*(_SEVERITY[risk] for risk in risks), min(1.0, round(gap_max, 2))],
                }
            ],
        },
    }
