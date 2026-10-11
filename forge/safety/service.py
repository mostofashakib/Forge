"""One gate for every dataset row: threat rules, then scrubbing, then the judge.

The order matters. Threat rules read raw text, so a payload cannot hide behind
a pseudonym. The optional LLM judge reads scrubbed text, so PII never reaches
a model provider.
"""
from __future__ import annotations

import re
from collections import Counter

from forge.safety.classifier import SafetyClassifier
from forge.safety.ner import NameDetector
from forge.safety.pii import detect_pii
from forge.safety.pseudonymizer import PSEUDONYM_PATTERN, Pseudonymizer
from forge.safety.secrets import REDACTED_SECRET, detect_secrets
from forge.safety.threats import detect_threats
from forge.safety.types import Finding, ScreenResult, Span

# Text this service already wrote. Spans touching it are left alone, so
# screening a row twice is a no-op.
_SCRUBBED = re.compile(PSEUDONYM_PATTERN.pattern + "|" + re.escape(REDACTED_SECRET))


class DatasetSafetyService:
    def __init__(
        self,
        pseudonymizer: Pseudonymizer,
        name_detector: NameDetector,
        classifier: SafetyClassifier | None = None,
    ) -> None:
        self._pseudonymizer = pseudonymizer
        self._names = name_detector
        self._classifier = classifier

    def screen(self, record: object) -> ScreenResult:
        """A scrubbed copy of `record`, kept only when nothing flags it."""
        texts = list(_strings(record))
        # Details quote raw text, so they are scrubbed before anyone stores them.
        findings = [
            Finding(finding.rule, self._scrub(finding.detail, Counter()))
            for text in texts
            for finding in detect_threats(text)
        ]
        replacements: Counter[str] = Counter()
        scrubbed = _map_strings(record, lambda text: self._scrub(text, replacements))
        if not findings and self._classifier is not None:
            scrubbed_text = "\n".join(_strings(scrubbed))
            if scrubbed_text:
                findings = self._classifier.classify(scrubbed_text)
        return ScreenResult(scrubbed, not findings, findings, dict(replacements))

    def _scrub(self, text: str, replacements: Counter[str]) -> str:
        spans = _resolve(detect_secrets(text) + detect_pii(text) + self._names.detect(text), text)
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            text = text[:span.start] + self._replacement(span) + text[span.end:]
            replacements[span.kind] += 1
        return text

    def _replacement(self, span: Span) -> str:
        if span.kind == "secret":
            return REDACTED_SECRET
        return self._pseudonymizer.replace(span.kind, span.value)


def _resolve(spans: list[Span], text: str) -> list[Span]:
    """Non-overlapping spans: longest first, a secret wins a tie."""
    taken = [(m.start(), m.end()) for m in _SCRUBBED.finditer(text)]
    chosen: list[Span] = []
    for span in sorted(spans, key=lambda s: (-(s.end - s.start), s.kind != "secret", s.start)):
        if all(span.end <= start or span.start >= end for start, end in taken):
            chosen.append(span)
            taken.append((span.start, span.end))
    return chosen


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _map_strings(value: object, transform) -> object:
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_strings(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_strings(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_strings(item, transform) for item in value)
    return value
