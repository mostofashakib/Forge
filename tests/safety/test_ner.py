"""Person names found by GLiNER2 become spans the service can pseudonymize."""
import pytest

from forge.safety.ner import DEFAULT_NER_MODEL, DEFAULT_NER_THRESHOLD, GlinerNameDetector, load_gliner


class StubModel:
    def __init__(self, people_by_text):
        self.people_by_text = people_by_text
        self.calls = []

    def batch_extract_entities_long(self, texts, entity_types, threshold, include_spans):
        self.calls.append((list(texts), entity_types, threshold, include_spans))
        return [{"entities": {"full_name": self.people_by_text.get(text, []), "first_name": []}} for text in texts]


def test_model_output_becomes_person_spans_in_one_batched_call():
    first, second = "Hi Sarah Chen", "ping Marcus"
    model = StubModel({
        first: [{"text": "Sarah Chen", "start": 3, "end": 13}],
        second: [{"text": "Marcus", "start": 5, "end": 11}],
    })
    spans = GlinerNameDetector(model, threshold=0.6).detect_many([first, second])

    assert [[(s.kind, s.value) for s in found] for found in spans] == [
        [("person", "Sarah Chen")],
        [("person", "Marcus")],
    ]
    assert model.calls == [([first, second], ["full_name", "first_name", "last_name"], 0.6, True)]


def test_every_name_label_maps_to_the_person_kind():
    class SplitLabels:
        def batch_extract_entities_long(self, texts, entity_types, threshold, include_spans):
            return [{"entities": {
                "first_name": [{"text": "Priya", "start": 0, "end": 5}],
                "last_name": [{"text": "Nair", "start": 6, "end": 10}],
            }}]
    (spans,) = GlinerNameDetector(SplitLabels(), threshold=0.5).detect_many(["Priya Nair"])
    assert {(s.kind, s.value) for s in spans} == {("person", "Priya"), ("person", "Nair")}


@pytest.mark.parametrize("text", ["", "   ", "12345", "{}", "0.5, 1.0"])
def test_text_without_letters_skips_the_model_but_keeps_its_slot(text):
    model = StubModel({"Hi Bob": [{"text": "Bob", "start": 3, "end": 6}]})
    spans = GlinerNameDetector(model, threshold=0.5).detect_many([text, "Hi Bob"])
    assert [[s.value for s in found] for found in spans] == [[], ["Bob"]]
    assert model.calls[0][0] == ["Hi Bob"]


def test_an_all_skipped_batch_never_calls_the_model():
    model = StubModel({})
    assert GlinerNameDetector(model, threshold=0.5).detect_many(["1", "2"]) == [[], []]
    assert model.calls == []


def test_spans_with_digits_are_rejected_as_names():
    # The real model tags severity tags like "P0" as people at 0.71.
    text = "[ALERT] P0 for Dana"
    model = StubModel({text: [{"text": "P0", "start": 8, "end": 10}, {"text": "Dana", "start": 15, "end": 19}]})
    (spans,) = GlinerNameDetector(model, threshold=0.5).detect_many([text])
    assert [s.value for s in spans] == ["Dana"]


def _weights_cached(model_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(model_id, "model.safetensors"), str)


@pytest.mark.skipif(not _weights_cached(DEFAULT_NER_MODEL), reason="GLiNER2-PII weights not in the HF cache")
def test_real_model_finds_names_in_trace_style_text():
    detector = GlinerNameDetector(load_gliner(DEFAULT_NER_MODEL), threshold=DEFAULT_NER_THRESHOLD)
    message = "Hi Sarah Chen, please forward the Q3 report to Marcus Webb before Friday. Thanks, Priya Nair"
    email, command, alert, pronouns = detector.detect_many([
        message,
        "$ close_ticket --id 42 --status resolved",
        "[ALERT] P0 — Production DB latency spike (p99 > 8s)",
        "I told you the sales team would ask me, and everyone agreed with her.",
    ])
    # A full name may come back whole or as first and last name. Either way
    # every character of it must be covered.
    covered = {index for span in email for index in range(span.start, span.end)}
    for name in ("Sarah Chen", "Marcus Webb", "Priya Nair"):
        start = message.index(name)
        assert {i for i in range(start, start + len(name)) if message[i] != " "} <= covered, name
    assert command == []
    assert alert == []
    # The `person` label tags pronouns and role words. The name labels must not.
    assert pronouns == []
