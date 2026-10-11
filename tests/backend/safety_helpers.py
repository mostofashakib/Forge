"""Test doubles for the dataset safety gate: a real service with a stub NER model."""
from __future__ import annotations

from pathlib import Path

from backend.app.services.export_writers.gate import DatasetGate
from forge.safety import config as safety_config
from forge.safety.ner import GlinerNameDetector
from forge.safety.pseudonymizer import Pseudonymizer
from forge.safety.service import DatasetSafetyService

TEST_SCRUB_KEY = "test-scrub-key-for-unit-tests-only"


class SubstringNameModel:
    """Stands in for GLiNER2: tags each listed name wherever it appears."""

    def __init__(self, *names: str) -> None:
        self.names = names

    def batch_extract_entities_long(self, texts, entity_types, threshold, include_spans):
        return [{"entities": {"person": self._people(text)}} for text in texts]

    def _people(self, text):
        people = []
        for name in self.names:
            start = text.find(name)
            while start != -1:
                people.append({"text": name, "start": start, "end": start + len(name)})
                start = text.find(name, start + 1)
        return people


def make_gate(out_dir: Path, *names: str) -> DatasetGate:
    service = DatasetSafetyService(
        Pseudonymizer(TEST_SCRUB_KEY.encode()),
        GlinerNameDetector(SubstringNameModel(*names), threshold=0.5),
    )
    return DatasetGate(service, out_dir / "safety_quarantine.jsonl", mode="rules")


def use_safety_env(monkeypatch, *names: str) -> None:
    """Configure the real env-driven gate without downloading the NER model."""
    monkeypatch.setenv("FORGE_SCRUB_KEY", TEST_SCRUB_KEY)
    monkeypatch.delenv("FORGE_SAFETY_LLM", raising=False)
    monkeypatch.setattr(safety_config, "load_gliner", lambda _model_id: SubstringNameModel(*names))
