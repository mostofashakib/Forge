"""Adult-content detection with a small local text classifier.

The classifier reads at most 512 tokens, so each text is cut into overlapping
windows and scored on its most explicit window. Every window of a row goes to
the model in one batched call.
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
        """One `adult` finding per text that scores at or above the threshold."""


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
        windows: list[str] = []
        owners: list[int] = []
        for index, text in enumerate(texts):
            if _HAS_LETTER.search(text):
                for window in _windows(text):
                    windows.append(window)
                    owners.append(index)
        if not windows:
            return []
        results = self._classifier(windows, truncation=True, max_length=_MAX_TOKENS, batch_size=_BATCH_SIZE)
        best: dict[int, float] = {}
        for owner, scores in zip(owners, results):
            score = next(s["score"] for s in scores if s["label"].lower() == _LABEL)
            best[owner] = max(best.get(owner, 0.0), score)
        return [
            Finding("adult", f"score {score:.2f}: {texts[owner][:_DETAIL_CHARS]}")
            for owner, score in sorted(best.items())
            if score >= self._threshold
        ]


def _windows(text: str) -> list[str]:
    step = WINDOW_CHARS - _OVERLAP_CHARS
    return [text[start:start + WINDOW_CHARS] for start in range(0, max(len(text) - _OVERLAP_CHARS, 1), step)]
