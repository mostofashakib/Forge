"""Person-name detection with GLiNER2-PII, run locally.

Names have no regex shape, so a model finds them. The model loads once per
process. Every string in a row goes to the model in one batched call, and
long text is chunked by the model, so a full trajectory is scanned end to end
rather than truncated.
"""
from __future__ import annotations

import re
from functools import lru_cache
from collections.abc import Sequence
from typing import Any, Protocol

from forge.safety.types import Span

DEFAULT_NER_MODEL = "fastino/gliner2-privacy-filter-PII-multi"
# Measured on the premade Gmail/Slack seeds: 0.8 keeps every real name there,
# 0.9 starts missing them, and lower values add product names.
DEFAULT_NER_THRESHOLD = 0.8
# The model's broad `person` label also tags pronouns and role words
# ("I", "you", "sales team"). The name labels do not.
_LABELS = ["full_name", "first_name", "last_name"]
_KIND = "person"
_HAS_LETTER = re.compile(r"[^\W\d_]")
# Names never contain digits. The model tags severity tags like "P0" otherwise.
_HAS_DIGIT = re.compile(r"\d")


class NameDetector(Protocol):
    def detect_many(self, texts: Sequence[str]) -> list[list[Span]]:
        """Person spans for each text, in the same order as `texts`."""


@lru_cache(maxsize=None)
def load_gliner(model_id: str) -> Any:
    """Load a GLiNER2 checkpoint once per process."""
    from gliner2 import GLiNER2

    return GLiNER2.from_pretrained(model_id)


class GlinerNameDetector:
    def __init__(self, model: Any, threshold: float) -> None:
        self._model = model
        self._threshold = threshold

    def detect_many(self, texts: Sequence[str]) -> list[list[Span]]:
        found: list[list[Span]] = [[] for _ in texts]
        # Text with no letters cannot hold a name, so it skips the model.
        indexes = [index for index, text in enumerate(texts) if _HAS_LETTER.search(text)]
        if not indexes:
            return found
        results = self._model.batch_extract_entities_long(
            [texts[index] for index in indexes], _LABELS, threshold=self._threshold, include_spans=True
        )
        for index, result in zip(indexes, results):
            found[index] = [
                Span(entity["start"], entity["end"], _KIND, entity["text"])
                for entities in result["entities"].values()
                for entity in entities
                if not _HAS_DIGIT.search(entity["text"])
            ]
        return found
