"""Adult-content detection with a small local text classifier.

A row's strings are joined and scored as one text. Scored alone, a short
field like the folder name "trash" reads as explicit. The classifier reads at
most 512 tokens, so the text is cut into overlapping windows, scored in one
batched call, and judged by its most explicit window.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

from forge.safety.types import Finding

DEFAULT_ADULT_MODEL = "eliasalbouzidi/distilbert-nsfw-text-classifier"
DEFAULT_ADULT_THRESHOLD = 0.9
# About 375 tokens of English, inside the model's 512-token limit.
WINDOW_CHARS = 1500
_OVERLAP_CHARS = 200
_MAX_TOKENS = 512
_BATCH_SIZE = 16
_LABEL = "nsfw"
_DETAIL_CHARS = 80
_HAS_LETTER = re.compile(r"[^\W\d_]")


class AdultDetector(Protocol):
    def detect(self, texts: Sequence[str]) -> list[Finding]:
        """One `adult` finding when the row made of `texts` scores at or above the threshold."""


@lru_cache(maxsize=None)
def load_adult_pipeline(model_id: str) -> Any:
    """Load a text-classification pipeline once per process."""
    from transformers import pipeline

    return pipeline("text-classification", model=model_id, top_k=None)


class DistilBertAdultDetector:
    def __init__(self, classifier: Any, threshold: float) -> None:
        labels = {str(label).lower() for label in classifier.model.config.id2label.values()}
        if _LABEL not in labels:
            # Scoring a missing label as 0 would let every row through.
            raise ValueError(f"adult classifier has no {_LABEL!r} label, only {sorted(labels)}")
        self._classifier = classifier
        self._threshold = threshold

    def detect(self, texts: Sequence[str]) -> list[Finding]:
        row = "\n".join(text for text in texts if _HAS_LETTER.search(text))
        if not row:
            return []
        windows = _windows(row)
        results = self._classifier(windows, truncation=True, max_length=_MAX_TOKENS, batch_size=_BATCH_SIZE)
        score, window = max(
            (next(s["score"] for s in scores if s["label"].lower() == _LABEL), window)
            for scores, window in zip(results, windows)
        )
        if score < self._threshold:
            return []
        return [Finding("adult", f"score {score:.2f}: {window[:_DETAIL_CHARS]}")]


def _windows(text: str) -> list[str]:
    step = WINDOW_CHARS - _OVERLAP_CHARS
    return [text[start:start + WINDOW_CHARS] for start in range(0, max(len(text) - _OVERLAP_CHARS, 1), step)]
