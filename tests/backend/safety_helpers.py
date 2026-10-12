"""Test doubles for the dataset safety gate: a real service with stub models."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend.app.services.export_writers.gate import DatasetGate
from forge.safety import config as safety_config
from forge.safety.adult import DistilBertAdultDetector
from forge.safety.ner import GlinerNameDetector
from forge.safety.pseudonymizer import Pseudonymizer
from forge.safety.service import DatasetSafetyService, PiiScrubbing
from forge.safety.types import DataSource

TEST_SCRUB_KEY = "test-scrub-key-for-unit-tests-only"
# Text containing this marker scores as explicit with the stub adult classifier.
ADULT_MARKER = "EXPLICIT_ADULT_TEXT"


class SubstringNameModel:
    """Stands in for GLiNER2: tags each listed name wherever it appears."""

    def __init__(self, *names: str) -> None:
        self.names = names

    def batch_extract_entities_long(self, texts, entity_types, threshold, include_spans):
        return [{"entities": {"full_name": self._people(text)}} for text in texts]

    def _people(self, text):
        people = []
        for name in self.names:
            start = text.find(name)
            while start != -1:
                people.append({"text": name, "start": start, "end": start + len(name)})
                start = text.find(name, start + 1)
        return people


class MarkerAdultPipeline:
    """Stands in for the NSFW classifier: explicit only when the marker appears."""

    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "safe", 1: "nsfw"}))

    def __call__(self, windows, truncation, max_length, batch_size):
        return [[{"label": "nsfw", "score": 1.0 if ADULT_MARKER in w else 0.0}] for w in windows]


def make_gate(out_dir: Path, *names: str, source: DataSource = DataSource.INTERNET) -> DatasetGate:
    adult = DistilBertAdultDetector(MarkerAdultPipeline(), threshold=0.9)
    pii = (
        PiiScrubbing(Pseudonymizer(TEST_SCRUB_KEY.encode()), GlinerNameDetector(SubstringNameModel(*names), 0.5))
        if source.scrubs_pii else None
    )
    return DatasetGate(DatasetSafetyService(adult, pii), out_dir / "safety_quarantine.jsonl", "rules", source)


def use_safety_env(monkeypatch, *names: str) -> None:
    """Configure the real env-driven gate without downloading any model."""
    monkeypatch.setenv("FORGE_SCRUB_KEY", TEST_SCRUB_KEY)
    monkeypatch.delenv("FORGE_SAFETY_LLM", raising=False)
    monkeypatch.setattr(safety_config, "load_gliner", lambda _model_id: SubstringNameModel(*names))
    monkeypatch.setattr(safety_config, "load_adult_pipeline", lambda _model_id: MarkerAdultPipeline())
