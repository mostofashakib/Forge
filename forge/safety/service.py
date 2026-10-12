"""One gate for every dataset row: harm checks, then scrubbing, then the judge.

The order matters. Threat and copyright rules read raw text, so a payload
cannot hide behind a pseudonym. The optional LLM judge reads scrubbed text, so
PII never reaches a model provider.

Secrets are redacted for every source. PII is scrubbed only when the service
gets a `PiiScrubbing`, which pipelines pass for internet and human data.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from forge.safety.adult import AdultDetector
from forge.safety.classifier import SafetyClassifier
from forge.safety.copyright import detect_copyright
from forge.safety.ner import NameDetector
from forge.safety.pii import detect_pii
from forge.safety.pseudonymizer import PSEUDONYM_PATTERN, Pseudonymizer
from forge.safety.secrets import REDACTED_SECRET, detect_secrets
from forge.safety.threats import detect_threats
from forge.safety.types import Finding, ScreenResult, Span

# Text this service already wrote. Spans touching it are left alone, so
# screening a row twice is a no-op.
_SCRUBBED = re.compile(PSEUDONYM_PATTERN.pattern + "|" + re.escape(REDACTED_SECRET))


@dataclass(frozen=True)
class PiiScrubbing:
    """What PII scrubbing needs: keyed pseudonyms, and a name detector unless names stay.

    Source code passes `names=None`: the name model reads identifiers such as
    a package name as people, and rewriting them breaks the code.
    """

    pseudonymizer: Pseudonymizer
    names: NameDetector | None


class DatasetSafetyService:
    def __init__(
        self,
        adult: AdultDetector,
        pii: PiiScrubbing | None = None,
        classifier: SafetyClassifier | None = None,
    ) -> None:
        self._adult = adult
        self._pii = pii
        self._classifier = classifier

    def screen(self, record: object) -> ScreenResult:
        """A scrubbed copy of `record`, kept only when nothing flags it."""
        texts = list(_strings(record))
        names = self._detect_names(texts) if self._pii is not None and self._pii.names is not None else None
        rule_findings = self._scrub_details(
            [f for text in texts for f in detect_threats(text) + detect_copyright(text)], names
        )
        adult_findings = self._scrub_details(self._adult.detect(texts), names)
        replacements: Counter[str] = Counter()
        scrubbed = _map_strings(record, lambda text: self._scrub(text, names, replacements))
        findings = rule_findings + adult_findings
        scrubbed_text = "\n".join(_strings(scrubbed))
        # The judge sees every row no rule blocked. It also has the last word on
        # rows only the adult classifier flagged, which misreads sex education.
        if not rule_findings and self._classifier is not None and scrubbed_text:
            findings = self._classifier.classify(scrubbed_text)
        return ScreenResult(scrubbed, not findings, findings, dict(replacements))

    def _scrub_details(self, findings: list[Finding], names: re.Pattern[str] | None) -> list[Finding]:
        """Details quote raw text, so they are scrubbed before anyone stores them."""
        return [Finding(f.rule, self._scrub(f.detail, names, Counter())) for f in findings]

    def _detect_names(self, texts: list[str]) -> re.Pattern[str] | None:
        """One pattern matching every name the model found anywhere in the row.

        The model can tag a name once and miss its later mentions, so each
        found name is replaced at every whole-word, same-case occurrence.
        """
        unique = list(dict.fromkeys(texts))
        found = {span.value for spans in self._pii.names.detect_many(unique) for span in spans}
        # A one-letter "name" would replace every "C" in "Option C".
        found = {name for name in found if len(name.strip()) >= 2}
        if not found:
            return None
        alternatives = "|".join(re.escape(name) for name in sorted(found, key=len, reverse=True))
        return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")

    def _scrub(self, text: str, names: re.Pattern[str] | None, replacements: Counter[str]) -> str:
        spans = detect_secrets(text)
        if self._pii is not None:
            spans += detect_pii(text)
        if names is not None:
            spans += [Span(m.start(), m.end(), "person", m.group()) for m in names.finditer(text)]
        for span in sorted(_resolve(spans, text), key=lambda s: s.start, reverse=True):
            text = text[:span.start] + self._replacement(span) + text[span.end:]
            replacements[span.kind] += 1
        return text

    def _replacement(self, span: Span) -> str:
        if span.kind == "secret":
            return REDACTED_SECRET
        return self._pii.pseudonymizer.replace(span.kind, span.value)


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
