from datetime import datetime, timezone
import pytest

from forge.taskfactory.contamination import (
    NGramOverlapDetector,
    CutoffContaminationChecker,
    ParaphraseRobustnessChecker,
    TaskContaminationVerifier,
    paraphrase_draft_text,
)
from forge.taskfactory.schemas import TaskDraft, TaskSeed


def test_ngram_overlap_detector():
    detector = NGramOverlapDetector(n=4, threshold=0.5, max_consecutive_threshold=6)
    training_data = [
        "the quick brown fox jumps over the lazy dog and runs away",
        "system prompt for model alignment and safety evaluation",
    ]

    # Test item with high overlap
    contaminated_item = "the quick brown fox jumps over the lazy dog completely"
    res1 = detector.check_overlap(contaminated_item, training_data)
    assert res1.is_contaminated
    assert res1.overlap_ratio >= 0.5
    assert res1.matching_ngrams_count > 0

    # Test item with no overlap
    clean_item = "completely novel scientific calculation of molecular orbital configurations"
    res2 = detector.check_overlap(clean_item, training_data)
    assert not res2.is_contaminated
    assert res2.overlap_ratio == 0.0


def test_cutoff_contamination_checker():
    checker = CutoffContaminationChecker(default_cutoff_date="2024-04-01")

    # Item published after cutoff (strictly clean)
    res_post = checker.evaluate_item(item_date="2024-05-15")
    assert res_post.is_post_cutoff
    assert not res_post.is_contaminated

    # Item published before cutoff when strict post-cutoff is required
    res_pre = checker.evaluate_item(item_date="2023-11-01", require_post_cutoff=True)
    assert not res_pre.is_post_cutoff
    assert res_pre.is_contaminated

    # Unknown date with require_post_cutoff
    res_unknown = checker.evaluate_item(item_date=None, require_post_cutoff=True)
    assert not res_unknown.is_post_cutoff
    assert res_unknown.is_contaminated


def test_paraphrase_robustness_checker():
    checker = ParaphraseRobustnessChecker(drop_threshold=0.35)

    # Memorization scenario: high original score, collapses on paraphrase
    res_mem = checker.compare_performance(original_score=0.95, paraphrased_score=0.40)
    assert res_mem.is_contaminated
    assert res_mem.performance_drop == pytest.approx(0.55)

    # Robust generalization: scores are close
    res_robust = checker.compare_performance(original_score=0.88, paraphrased_score=0.82)
    assert not res_robust.is_contaminated
    assert res_robust.performance_drop == pytest.approx(0.06)


def test_unified_task_contamination_verifier():
    verifier = TaskContaminationVerifier(
        training_corpus=["configure database connection pool with maximum twenty clients"],
        cutoff_date="2024-04-01",
        ngram_threshold=0.3,
        drop_threshold=0.3,
    )

    draft_clean = TaskDraft(
        slot=1,
        title="Novel API Handler",
        objective="Implement an HTTP endpoint returning status code 200",
        seed=TaskSeed(),
        golden=[],
        checks=[],
    )

    report_clean = verifier.verify_draft(draft_clean)
    assert not report_clean.is_contaminated
    assert len(report_clean.reasons) == 0

    draft_contaminated = TaskDraft(
        slot=2,
        title="Database Setup",
        objective="configure database connection pool with maximum twenty clients",
        seed=TaskSeed(),
        golden=[],
        checks=[],
    )

    report_contam = verifier.verify_draft(draft_contaminated)
    assert report_contam.is_contaminated
    assert any("n-gram" in r for r in report_contam.reasons)


def test_paraphrase_draft_text():
    orig = "verify that all records match and create a new session"
    para = paraphrase_draft_text(orig)
    assert "ensure that" in para
    assert "generate a new" in para
