"""Contamination detection for synthetic task verification.

Checks synthetic task contamination in three ways (FR2):
1. N-gram overlap tests between training data and test items.
2. Evaluation of items published after the model's training cutoff.
3. Performance comparison on original items vs paraphrased / rewritten versions.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Sequence
from pydantic import BaseModel, Field

from forge.taskfactory.schemas import TaskDraft


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def _extract_ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """Extract set of n-grams from a sequence of tokens."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class NGramOverlapResult(BaseModel):
    n: int
    overlap_ratio: float
    max_consecutive_matching_tokens: int
    threshold: float
    is_contaminated: bool
    matching_ngrams_count: int
    total_test_ngrams: int
    details: str = ""


class CutoffEvaluationResult(BaseModel):
    item_date: str | None
    cutoff_date: str | None
    is_post_cutoff: bool
    is_contaminated: bool
    details: str = ""


class ParaphraseComparisonResult(BaseModel):
    original_score: float
    paraphrased_score: float
    performance_drop: float
    drop_threshold: float
    is_contaminated: bool
    details: str = ""


class ContaminationReport(BaseModel):
    is_contaminated: bool
    reasons: list[str] = Field(default_factory=list)
    ngram_result: NGramOverlapResult | None = None
    cutoff_result: CutoffEvaluationResult | None = None
    paraphrase_result: ParaphraseComparisonResult | None = None


# ---------------------------------------------------------------------------
# 1. N-gram Overlap Detection
# ---------------------------------------------------------------------------

class NGramOverlapDetector:
    """Checks n-gram overlap between reference/training corpora and test items."""

    def __init__(self, n: int = 8, threshold: float = 0.4, max_consecutive_threshold: int = 13):
        self.n = n
        self.threshold = threshold
        self.max_consecutive_threshold = max_consecutive_threshold

    def check_overlap(
        self,
        test_text: str,
        training_corpus: Sequence[str] | str,
        n: int | None = None,
        threshold: float | None = None,
    ) -> NGramOverlapResult:
        n_val = n or self.n
        thresh = threshold if threshold is not None else self.threshold

        test_tokens = _tokenize(test_text)
        if len(test_tokens) < n_val:
            return NGramOverlapResult(
                n=n_val,
                overlap_ratio=0.0,
                max_consecutive_matching_tokens=0,
                threshold=thresh,
                is_contaminated=False,
                matching_ngrams_count=0,
                total_test_ngrams=0,
                details=f"Test text too short ({len(test_tokens)} tokens) for {n_val}-gram analysis.",
            )

        test_ngrams = _extract_ngrams(test_tokens, n_val)
        if isinstance(training_corpus, str):
            corpus_docs = [training_corpus]
        else:
            corpus_docs = list(training_corpus)

        corpus_ngrams: set[tuple[str, ...]] = set()
        for doc in corpus_docs:
            doc_tokens = _tokenize(doc)
            corpus_ngrams.update(_extract_ngrams(doc_tokens, n_val))

        matching_ngrams = test_ngrams.intersection(corpus_ngrams)
        overlap_ratio = len(matching_ngrams) / len(test_ngrams) if test_ngrams else 0.0

        # Calculate max consecutive matching tokens
        max_consecutive = 0
        current_consecutive = 0
        for i in range(len(test_tokens) - n_val + 1):
            gram = tuple(test_tokens[i : i + n_val])
            if gram in corpus_ngrams:
                current_consecutive = (current_consecutive + 1) if current_consecutive else n_val
                if current_consecutive > max_consecutive:
                    max_consecutive = current_consecutive
            else:
                current_consecutive = 0

        is_contaminated = (
            overlap_ratio >= thresh
            or max_consecutive >= self.max_consecutive_threshold
        )

        details = (
            f"Overlap ratio: {overlap_ratio:.3f} (thresh={thresh:.2f}), "
            f"Matching {n_val}-grams: {len(matching_ngrams)}/{len(test_ngrams)}, "
            f"Max consecutive matching tokens: {max_consecutive}."
        )

        return NGramOverlapResult(
            n=n_val,
            overlap_ratio=overlap_ratio,
            max_consecutive_matching_tokens=max_consecutive,
            threshold=thresh,
            is_contaminated=is_contaminated,
            matching_ngrams_count=len(matching_ngrams),
            total_test_ngrams=len(test_ngrams),
            details=details,
        )


# ---------------------------------------------------------------------------
# 2. Training Cutoff Evaluation
# ---------------------------------------------------------------------------

def _parse_datetime(date_val: str | datetime | None) -> datetime | None:
    if date_val is None:
        return None
    if isinstance(date_val, datetime):
        return date_val if date_val.tzinfo else date_val.replace(tzinfo=timezone.utc)
    try:
        # Support ISO 8601 or YYYY-MM-DD
        dt = datetime.fromisoformat(date_val.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(date_val, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


class CutoffContaminationChecker:
    """Evaluates items against the model's training cutoff date."""

    def __init__(self, default_cutoff_date: str | datetime = "2024-04-01"):
        self.default_cutoff_date = _parse_datetime(default_cutoff_date)

    def evaluate_item(
        self,
        item_date: str | datetime | None,
        cutoff_date: str | datetime | None = None,
        require_post_cutoff: bool = False,
    ) -> CutoffEvaluationResult:
        c_dt = _parse_datetime(cutoff_date) or self.default_cutoff_date
        i_dt = _parse_datetime(item_date)

        c_str = c_dt.isoformat() if c_dt else None
        i_str = i_dt.isoformat() if i_dt else None

        if not i_dt:
            # Date unknown: if post-cutoff is strictly required, it fails
            is_contaminated = require_post_cutoff
            details = (
                "Item publication date is unknown. "
                + ("Marked contaminated because post-cutoff verification was required." if require_post_cutoff else "Unverified date allowed.")
            )
            return CutoffEvaluationResult(
                item_date=None,
                cutoff_date=c_str,
                is_post_cutoff=False,
                is_contaminated=is_contaminated,
                details=details,
            )

        if not c_dt:
            return CutoffEvaluationResult(
                item_date=i_str,
                cutoff_date=None,
                is_post_cutoff=True,
                is_contaminated=False,
                details="No training cutoff date specified.",
            )

        is_post_cutoff = i_dt > c_dt
        # Contaminated if strictly before or equal to cutoff when strict evaluation is applied
        is_contaminated = require_post_cutoff and not is_post_cutoff

        details = (
            f"Item date {i_str} is {'AFTER' if is_post_cutoff else 'ON OR BEFORE'} cutoff {c_str}. "
            + ("Contaminated (pre-cutoff)." if is_contaminated else "Passed cutoff check.")
        )

        return CutoffEvaluationResult(
            item_date=i_str,
            cutoff_date=c_str,
            is_post_cutoff=is_post_cutoff,
            is_contaminated=is_contaminated,
            details=details,
        )


# ---------------------------------------------------------------------------
# 3. Paraphrase Performance Comparison
# ---------------------------------------------------------------------------

class ParaphraseRobustnessChecker:
    """Compares model performance on original items against paraphrased versions.

    A sharp drop in model performance when evaluated on semantically equivalent
    paraphrased tasks indicates surface memorization (contamination).
    """

    def __init__(self, drop_threshold: float = 0.35):
        self.drop_threshold = drop_threshold

    def compare_performance(
        self,
        original_score: float,
        paraphrased_score: float,
        drop_threshold: float | None = None,
    ) -> ParaphraseComparisonResult:
        threshold = drop_threshold if drop_threshold is not None else self.drop_threshold
        # Performance drop: high original score but model collapses on paraphrase
        drop = max(0.0, original_score - paraphrased_score)
        is_contaminated = (drop >= threshold) and (original_score > 0.4)

        details = (
            f"Original score: {original_score:.3f}, Paraphrased score: {paraphrased_score:.3f}, "
            f"Drop: {drop:.3f} (threshold: {threshold:.3f}). "
            + ("Contaminated (memorization detected via paraphrase divergence)." if is_contaminated else "Passed paraphrase robustness check.")
        )

        return ParaphraseComparisonResult(
            original_score=original_score,
            paraphrased_score=paraphrased_score,
            performance_drop=drop,
            drop_threshold=threshold,
            is_contaminated=is_contaminated,
            details=details,
        )


# ---------------------------------------------------------------------------
# Unified Contamination Verifier
# ---------------------------------------------------------------------------

def paraphrase_draft_text(objective: str) -> str:
    """Generate a paraphrased version of an objective statement."""
    # Deterministic semantic rephrase for automated verification tests
    replacements = [
        (r"\bverify that\b", "ensure that"),
        (r"\bcheck if\b", "determine whether"),
        (r"\bcreate a\b", "generate a new"),
        (r"\bupdate the\b", "modify the existing"),
        (r"\bdelete the\b", "remove the"),
        (r"\bfind the\b", "locate the"),
        (r"\brun the\b", "execute the"),
        (r"\bconfigure\b", "set up"),
        (r"\bdisplay\b", "show"),
        (r"\bretrieve\b", "fetch"),
    ]
    paraphrased = objective
    for pattern, repl in replacements:
        paraphrased = re.sub(pattern, repl, paraphrased, flags=re.IGNORECASE)

    if paraphrased == objective:
        paraphrased = f"Please complete the following goal: {objective}"
    return paraphrased


class TaskContaminationVerifier:
    """Verifies synthetic tasks for data contamination across all 3 criteria."""

    def __init__(
        self,
        training_corpus: Sequence[str] | None = None,
        cutoff_date: str | datetime = "2024-04-01",
        ngram_threshold: float = 0.4,
        drop_threshold: float = 0.35,
    ):
        self.training_corpus = list(training_corpus or [])
        self.ngram_detector = NGramOverlapDetector(threshold=ngram_threshold)
        self.cutoff_checker = CutoffContaminationChecker(default_cutoff_date=cutoff_date)
        self.paraphrase_checker = ParaphraseRobustnessChecker(drop_threshold=drop_threshold)

    def verify_draft(
        self,
        draft: TaskDraft,
        training_corpus: Sequence[str] | None = None,
        item_date: str | datetime | None = None,
        original_score: float | None = None,
        paraphrased_score: float | None = None,
        require_post_cutoff: bool = False,
    ) -> ContaminationReport:
        corpus = training_corpus or self.training_corpus
        reasons: list[str] = []

        # 1. N-gram overlap check
        ngram_res: NGramOverlapResult | None = None
        if corpus:
            test_content = f"{draft.title} {draft.objective}"
            ngram_res = self.ngram_detector.check_overlap(test_content, corpus)
            if ngram_res.is_contaminated:
                reasons.append(f"n-gram overlap test failed: {ngram_res.details}")

        # 2. Cutoff evaluation
        cutoff_res: CutoffEvaluationResult | None = None
        if item_date is not None or require_post_cutoff:
            cutoff_res = self.cutoff_checker.evaluate_item(
                item_date=item_date,
                require_post_cutoff=require_post_cutoff,
            )
            if cutoff_res.is_contaminated:
                reasons.append(f"training cutoff evaluation failed: {cutoff_res.details}")

        # 3. Paraphrase performance comparison
        paraphrase_res: ParaphraseComparisonResult | None = None
        if original_score is not None and paraphrased_score is not None:
            paraphrase_res = self.paraphrase_checker.compare_performance(
                original_score=original_score,
                paraphrased_score=paraphrased_score,
            )
            if paraphrase_res.is_contaminated:
                reasons.append(f"paraphrase comparison failed: {paraphrase_res.details}")

        is_contaminated = len(reasons) > 0
        return ContaminationReport(
            is_contaminated=is_contaminated,
            reasons=reasons,
            ngram_result=ngram_res,
            cutoff_result=cutoff_res,
            paraphrase_result=paraphrase_res,
        )
