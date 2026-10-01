"""Tests for benchmark metrics: pass@1, pass@k, pass^k, Cohen's kappa and diagnostics (Feature Request 9)."""
import math
import pytest
from forge.benchmark.metrics import (
    TaskTrial,
    compute_cohens_kappa,
    compute_pass_at_k,
    compute_pass_pow_k,
    generate_benchmark_graphs,
)


def test_pass_at_k_and_pass_pow_k_computations():
    # When 0 passes, both are 0
    assert compute_pass_at_k(10, 0, 1) == 0.0
    assert compute_pass_pow_k(10, 0, 1) == 0.0

    # When all pass, both are 1.0
    assert compute_pass_at_k(10, 10, 1) == 1.0
    assert compute_pass_at_k(10, 10, 5) == 1.0
    assert compute_pass_pow_k(10, 10, 1) == 1.0
    assert compute_pass_pow_k(10, 10, 5) == 1.0

    # Pass@1 equals c / n
    assert math.isclose(compute_pass_at_k(10, 4, 1), 0.4)
    assert math.isclose(compute_pass_pow_k(10, 4, 1), 0.4)

    # For k > 1, pass@k >= pass@1 and pass^k <= pass@1
    pass_at_3 = compute_pass_at_k(10, 4, 3)
    pass_pow_3 = compute_pass_pow_k(10, 4, 3)
    assert pass_at_3 > 0.4
    assert pass_pow_3 < 0.4

    # If c < k, pass^k must be 0
    assert compute_pass_pow_k(10, 2, 3) == 0.0


def test_cohens_kappa_agreement():
    # Empty or unequal lengths
    assert compute_cohens_kappa([], []) == 0.0
    assert compute_cohens_kappa([True], [True, False]) == 0.0

    # Perfect agreement
    eval1 = [True, False, True, True, False, False, True, False]
    eval2 = [True, False, True, True, False, False, True, False]
    assert math.isclose(compute_cohens_kappa(eval1, eval2), 1.0)

    # Near chance agreement
    eval_a = [True, True, True, True, False, False, False, False]
    eval_b = [True, False, True, False, True, False, True, False]
    assert compute_cohens_kappa(eval_a, eval_b) < 0.2


def test_detect_reward_hacking_and_memorization_and_contamination():
    # 1. Hacking scenario: High pass@k but severe drop in pass^k / verifier disagreement
    hacking_trials = [
        TaskTrial(
            task_id=f"t_{i}",
            passed_samples=2,
            total_samples=10,
            verifier_passes=[True, True, False, False],
            ground_truth_passes=[False, False, False, False],
        )
        for i in range(10)
    ]
    graphs = generate_benchmark_graphs(hacking_trials, max_k=5)
    assert graphs.diagnostics["reward_hacking_risk"] in ("moderate", "high")
    assert "pass_curve" in graphs.chart_configs

    # 2. Memorization scenario: High success on original, collapse on paraphrased
    mem_trials = [
        TaskTrial(
            task_id=f"t_{i}",
            passed_samples=10,
            total_samples=10,
            paraphrased_passed_samples=0,
            paraphrased_total_samples=10,
        )
        for i in range(10)
    ]
    graphs_mem = generate_benchmark_graphs(mem_trials, max_k=5)
    assert graphs_mem.diagnostics["memorization_risk"] == "high"
    assert graphs_mem.diagnostics["memorization_performance_gap"] > 0.5

    # 3. Contamination scenario: Strong correlation between high ngram overlap and passes
    contam_trials = [
        TaskTrial(
            task_id=f"high_{i}",
            passed_samples=10,
            total_samples=10,
            ngram_overlap=0.85,
        )
        for i in range(5)
    ] + [
        TaskTrial(
            task_id=f"low_{i}",
            passed_samples=1,
            total_samples=10,
            ngram_overlap=0.1,
        )
        for i in range(5)
    ]
    graphs_contam = generate_benchmark_graphs(contam_trials, max_k=5)
    assert graphs_contam.diagnostics["contamination_risk"] == "high"


@pytest.mark.parametrize(("n", "c", "k"), [(5, -1, 1), (5, 6, 1), (5, 2, 0), (-1, 0, 1)])
def test_rejects_impossible_counts(n, c, k):
    with pytest.raises(ValueError):
        compute_pass_at_k(n=n, c=c, k=k)


def test_rejects_invalid_sample_count():
    # Negative path: passing k > n should raise ValueError or return 0
    with pytest.raises((ValueError, ZeroDivisionError)):
        compute_pass_at_k(n=0, c=0, k=1)


def test_graphs_refuse_to_rate_risk_without_any_trials():
    with pytest.raises(ValueError, match="no trials"):
        generate_benchmark_graphs([])
