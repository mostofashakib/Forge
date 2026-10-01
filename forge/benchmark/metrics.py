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
    """
    if n <= 0 or k <= 0:
        return 0.0
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
        return BenchmarkGraphData(
            k_labels=[1],
            pass_at_k=[0.0],
            pass_pow_k=[0.0],
            cohens_kappa=0.0,
            gap_pass_vs_pow=[0.0],
            diagnostics={
                "reward_hacking_risk": "low",
                "memorization_risk": "low",
                "contamination_risk": "low",
                "genuine_learning_index": 0.0,
            },
            chart_configs={},
        )

    # Determine feasible range of k
    min_total = min(t.total_samples for t in trials)
    limit_k = max_k if max_k is not None else min(min_total, 10)
    limit_k = max(1, limit_k)
    k_range = list(range(1, limit_k + 1))

    # Compute pass@k and pass^k curves across all tasks
    pass_at_k_series: list[float] = []
    pass_pow_k_series: list[float] = []
    gap_series: list[float] = []

    for k in k_range:
        scores_at_k = [
            compute_pass_at_k(t.total_samples, t.passed_samples, k) for t in trials
        ]
        scores_pow_k = [
            compute_pass_pow_k(t.total_samples, t.passed_samples, k) for t in trials
        ]
        mean_at_k = sum(scores_at_k) / len(scores_at_k)
        mean_pow_k = sum(scores_pow_k) / len(scores_pow_k)
        pass_at_k_series.append(round(mean_at_k, 4))
        pass_pow_k_series.append(round(mean_pow_k, 4))
        gap_series.append(round(mean_at_k - mean_pow_k, 4))

    # Compute Cohen's kappa agreement
    # Priority 1: verifier vs ground_truth passes
    # Priority 2: original vs paraphrased binary outcomes
    all_v_passes: list[bool] = []
    all_gt_passes: list[bool] = []
    for t in trials:
        if t.verifier_passes and t.ground_truth_passes:
            all_v_passes.extend(t.verifier_passes)
            all_gt_passes.extend(t.ground_truth_passes)

    orig_binary: list[bool] = []
    para_binary: list[bool] = []
    for t in trials:
        if t.paraphrased_total_samples is not None and t.paraphrased_passed_samples is not None:
            orig_binary.append(t.passed_samples > 0)
            para_binary.append(t.paraphrased_passed_samples > 0)

    if all_v_passes and len(all_v_passes) == len(all_gt_passes):
        kappa = compute_cohens_kappa(all_v_passes, all_gt_passes)
    elif orig_binary:
        kappa = compute_cohens_kappa(orig_binary, para_binary)
    else:
        # Measure inter-seed/trial consistency across tasks
        first_half = [t.passed_samples >= (t.total_samples / 2) for t in trials]
        second_half = [t.passed_samples > 0 for t in trials]
        kappa = compute_cohens_kappa(first_half, second_half)

    # Diagnostic Signals:
    # 1. Reward Hacking Risk:
    # High gap between pass@k and pass^k (> 0.5) OR verifier/ground truth kappa < 0.4
    pass1 = pass_at_k_series[0] if pass_at_k_series else 0.0
    pass_k_max = pass_at_k_series[-1] if pass_at_k_series else 0.0
    pass_pow_max = pass_pow_k_series[-1] if pass_pow_k_series else 0.0
    gap_max = pass_k_max - pass_pow_max

    if gap_max > 0.6 or (all_v_passes and kappa < 0.3):
        reward_hacking_risk = "high"
    elif gap_max > 0.35 or (all_v_passes and kappa < 0.6):
        reward_hacking_risk = "moderate"
    else:
        reward_hacking_risk = "low"

    # 2. Memorization Risk:
    # Large performance drop from original to paraphrased items
    mem_gap = 0.0
    if orig_binary and para_binary:
        orig_rate = sum(orig_binary) / len(orig_binary)
        para_rate = sum(para_binary) / len(para_binary)
        mem_gap = max(0.0, orig_rate - para_rate)

    if mem_gap > 0.3:
        memorization_risk = "high"
    elif mem_gap > 0.15:
        memorization_risk = "moderate"
    else:
        memorization_risk = "low"

    # 3. Contamination Risk:
    # Positive correlation between n-gram overlap and pass rate
    high_overlap_passes = [
        t.passed_samples / t.total_samples
        for t in trials
        if t.ngram_overlap is not None and t.ngram_overlap > 0.5
    ]
    low_overlap_passes = [
        t.passed_samples / t.total_samples
        for t in trials
        if t.ngram_overlap is not None and t.ngram_overlap <= 0.5
    ]
    if high_overlap_passes and low_overlap_passes:
        diff = (sum(high_overlap_passes) / len(high_overlap_passes)) - (
            sum(low_overlap_passes) / len(low_overlap_passes)
        )
        if diff > 0.35:
            contamination_risk = "high"
        elif diff > 0.15:
            contamination_risk = "moderate"
        else:
            contamination_risk = "low"
    else:
        contamination_risk = "low"

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

    # Chart configurations formatted for frontend graphs (Chart.js / JSON schema)
    chart_configs = {
        "pass_curve": {
            "type": "line",
            "title": "Pass@k vs Pass^k Robustness Curve",
            "xAxis": {"title": "k (Samples Drawn)", "categories": [f"k={k}" for k in k_range]},
            "yAxis": {"title": "Probability", "min": 0.0, "max": 1.0},
            "series": [
                {"name": "pass@k (At least 1 pass)", "data": pass_at_k_series, "color": "#10b981"},
                {"name": "pass^k (All k pass)", "data": pass_pow_k_series, "color": "#6366f1"},
                {"name": "Robustness Gap (pass@k - pass^k)", "data": gap_series, "color": "#f59e0b"},
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
            "categories": [
                "Reward Hacking",
                "Memorization",
                "Contamination",
                "Robustness Gap",
            ],
            "series": [
                {
                    "name": "Risk Severity",
                    "data": [
                        1.0 if reward_hacking_risk == "high" else (0.5 if reward_hacking_risk == "moderate" else 0.1),
                        1.0 if memorization_risk == "high" else (0.5 if memorization_risk == "moderate" else 0.1),
                        1.0 if contamination_risk == "high" else (0.5 if contamination_risk == "moderate" else 0.1),
                        min(1.0, round(gap_max, 2)),
                    ],
                }
            ],
        },
    }

    return BenchmarkGraphData(
        k_labels=k_range,
        pass_at_k=pass_at_k_series,
        pass_pow_k=pass_pow_k_series,
        cohens_kappa=round(kappa, 4),
        gap_pass_vs_pow=gap_series,
        diagnostics=diagnostics,
        chart_configs=chart_configs,
    )
