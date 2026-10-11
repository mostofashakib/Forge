"""Person names found by GLiNER2 become spans the service can pseudonymize."""
import pytest

from forge.safety.ner import DEFAULT_NER_MODEL, GlinerNameDetector, load_gliner


class StubModel:
    def __init__(self, people):
        self.people = people
        self.calls = []

    def extract_entities_long(self, text, entity_types, threshold, include_spans):
        self.calls.append((text, entity_types, threshold, include_spans))
        return {"entities": {"person": self.people}}


def test_model_output_becomes_person_spans():
    text = "Hi Sarah Chen, ping Marcus"
    model = StubModel([
        {"text": "Sarah Chen", "start": 3, "end": 13},
        {"text": "Marcus", "start": 20, "end": 26},
    ])
    spans = GlinerNameDetector(model, threshold=0.6).detect(text)

    assert [(s.kind, s.value, text[s.start:s.end]) for s in spans] == [
        ("person", "Sarah Chen", "Sarah Chen"),
        ("person", "Marcus", "Marcus"),
    ]
    assert model.calls == [(text, ["person"], 0.6, True)]


@pytest.mark.parametrize("text", ["", "   ", "12345", "{}", "0.5, 1.0"])
def test_text_without_letters_skips_the_model(text):
    model = StubModel([])
    assert GlinerNameDetector(model, threshold=0.5).detect(text) == []
    assert model.calls == []


def _weights_cached(model_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(model_id, "model.safetensors"), str)


@pytest.mark.skipif(not _weights_cached(DEFAULT_NER_MODEL), reason="GLiNER2-PII weights not in the HF cache")
def test_real_model_finds_names_in_trace_style_text():
    detector = GlinerNameDetector(load_gliner(DEFAULT_NER_MODEL), threshold=0.5)
    found = {span.value for span in detector.detect(
        "Hi Sarah Chen, please forward the Q3 report to Marcus Webb before Friday. Thanks, Priya"
    )}
    assert {"Sarah Chen", "Marcus Webb"} <= found
    assert detector.detect("$ close_ticket --id 42 --status resolved") == []
