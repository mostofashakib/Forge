"""Person-name detection with GLiNER2-PII, run locally.

Names have no regex shape, so a model finds them. The model loads once per
process. Long text is chunked by `extract_entities_long`, so a full
trajectory is scanned end to end rather than truncated.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Protocol

from forge.safety.types import Span

DEFAULT_NER_MODEL = "fastino/gliner2-privacy-filter-PII-multi"
_LABEL = "person"
_HAS_LETTER = re.compile(r"[^\W\d_]")


class NameDetector(Protocol):
    def detect(self, text: str) -> list[Span]: ...


@lru_cache(maxsize=None)
def load_gliner(model_id: str) -> Any:
    """Load a GLiNER2 checkpoint once per process."""
    from gliner2 import GLiNER2

    return GLiNER2.from_pretrained(model_id)


class GlinerNameDetector:
    def __init__(self, model: Any, threshold: float) -> None:
        self._model = model
        self._threshold = threshold

    def detect(self, text: str) -> list[Span]:
        if not _HAS_LETTER.search(text):
            return []
        result = self._model.extract_entities_long(
            text, [_LABEL], threshold=self._threshold, include_spans=True
        )
        return [
            Span(entity["start"], entity["end"], _LABEL, entity["text"])
            for entity in result["entities"].get(_LABEL, [])
        ]
