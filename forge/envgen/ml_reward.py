"""Traditional ML-based reward scorers — no LLM required.

Two strategies:
  - SentenceEmbeddingScorer: cosine similarity using sentence-transformers
  - NGramScorer: BLEU or ROUGE-L lexical overlap

Both expose .score(reference, candidate) -> float where 0.0 = no match
and 1.0 = perfect match. Imports are lazy so missing optional deps only
fail at call time, not at module import.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _embedding_model = SentenceTransformer(_MODEL_NAME)
        except ImportError:
            logger.warning(
                "[ml_reward] sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _embedding_model


class SentenceEmbeddingScorer:
    """Cosine similarity of sentence embeddings (all-MiniLM-L6-v2)."""

    def score(self, reference: str, candidate: str) -> float:
        if not reference.strip() or not candidate.strip():
            return 0.0
        model = _get_embedding_model()
        if model is None:
            return 0.0
        try:
            import numpy as np  # type: ignore
            embs = model.encode([reference, candidate], normalize_embeddings=True)
            raw = float(np.dot(embs[0], embs[1]))
            # cosine is in [-1, 1]; map to [0, 1]
            return max(0.0, min(1.0, (raw + 1.0) / 2.0))
        except Exception as exc:
            logger.warning("[ml_reward] embedding score failed: %s", exc)
            return 0.0


class NGramScorer:
    """BLEU or ROUGE-L/1/2 lexical overlap.

    metric options: "rougeL" (default), "rouge1", "rouge2", "bleu"
    """

    def __init__(self, metric: str = "rougeL") -> None:
        if metric not in ("rougeL", "rouge1", "rouge2", "bleu"):
            raise ValueError(f"Unknown metric: {metric!r}")
        self._metric = metric

    def score(self, reference: str, candidate: str) -> float:
        if not reference.strip() or not candidate.strip():
            return 0.0
        try:
            if self._metric.startswith("rouge"):
                return self._rouge_score(reference, candidate)
            return self._bleu_score(reference, candidate)
        except Exception as exc:
            logger.warning("[ml_reward] ngram score failed: %s", exc)
            return 0.0

    def _rouge_score(self, reference: str, candidate: str) -> float:
        from rouge_score import rouge_scorer as rs  # type: ignore
        metric_key = self._metric
        scorer = rs.RougeScorer([metric_key], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return float(getattr(scores[metric_key], "fmeasure", 0.0))

    def _bleu_score(self, reference: str, candidate: str) -> float:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
        ref_tokens = reference.lower().split()
        hyp_tokens = candidate.lower().split()
        if not hyp_tokens:
            return 0.0
        smoothing = SmoothingFunction().method1
        return float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing))


def build_scorer(method: str) -> SentenceEmbeddingScorer | NGramScorer | None:
    """Return the right scorer for a method string, or None for 'llm'."""
    if method == "embeddings":
        return SentenceEmbeddingScorer()
    if method == "rouge":
        return NGramScorer(metric="rougeL")
    if method == "bleu":
        return NGramScorer(metric="bleu")
    return None
