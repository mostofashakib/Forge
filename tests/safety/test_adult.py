"""The adult classifier scores every window of every text and flags explicit rows."""
from types import SimpleNamespace

import pytest

from forge.safety.adult import (
    DEFAULT_ADULT_MODEL,
    DEFAULT_ADULT_THRESHOLD,
    WINDOW_CHARS,
    DistilBertAdultDetector,
    load_adult_pipeline,
)


class StubPipeline:
    """Scores a window 0.99 when it contains a marker word, else 0.01."""

    def __init__(self, marker="EXPLICIT", labels=("safe", "nsfw")):
        self.marker = marker
        self.model = SimpleNamespace(config=SimpleNamespace(id2label=dict(enumerate(labels))))
        self.calls = []

    def __call__(self, windows, truncation, max_length, batch_size):
        self.calls.append(list(windows))
        assert truncation and max_length == 512
        return [
            [{"label": "nsfw", "score": 0.99 if self.marker in w else 0.01},
             {"label": "safe", "score": 0.01 if self.marker in w else 0.99}]
            for w in windows
        ]


def test_an_explicit_row_gets_one_finding_and_a_clean_row_none():
    detector = DistilBertAdultDetector(StubPipeline(), threshold=0.9)
    (finding,) = detector.detect(["EXPLICIT story", "more EXPLICIT text"])
    assert finding.rule == "adult" and finding.detail.startswith("score 0.99: ")
    assert detector.detect(["budget memo", "inbox"]) == []


def test_a_row_is_scored_as_one_text_so_short_fields_keep_their_context():
    # Alone, a folder name like "trash" scores 1.00 with the real model.
    pipe = StubPipeline()
    DistilBertAdultDetector(pipe, threshold=0.9).detect(["one", "trash", "three"])
    assert pipe.calls == [["one\ntrash\nthree"]]


def test_content_past_the_first_window_is_still_scanned():
    long_text = "x " * WINDOW_CHARS + "EXPLICIT"
    findings = DistilBertAdultDetector(StubPipeline(), threshold=0.9).detect([long_text])
    assert [f.rule for f in findings] == ["adult"]


def test_a_row_without_letters_skips_the_model():
    pipe = StubPipeline()
    assert DistilBertAdultDetector(pipe, threshold=0.9).detect(["", "42", "{}"]) == []
    assert DistilBertAdultDetector(pipe, threshold=0.9).detect([]) == []
    assert pipe.calls == []


def test_a_score_below_the_threshold_is_not_flagged():
    assert DistilBertAdultDetector(StubPipeline(), threshold=0.995).detect(["EXPLICIT"]) == []


def test_a_model_without_an_nsfw_label_is_rejected():
    with pytest.raises(ValueError, match="nsfw"):
        DistilBertAdultDetector(StubPipeline(labels=("LABEL_0", "LABEL_1")), threshold=0.9)


def _weights_cached(model_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


@pytest.mark.skipif(not _weights_cached(DEFAULT_ADULT_MODEL), reason="NSFW classifier weights not in the HF cache")
def test_real_model_flags_explicit_rows_and_passes_medical_code_and_short_fields():
    detector = DistilBertAdultDetector(load_adult_pipeline(DEFAULT_ADULT_MODEL), DEFAULT_ADULT_THRESHOLD)
    explicit = "She undressed him slowly and they had sex on the bed, moaning loudly."
    assert len(detector.detect(["inbox", explicit])) == 1
    for clean_row in (
        ["Schedule a breast cancer screening for the patient next Tuesday."],
        ["def strip_tags(html): return re.sub('<[^>]+>', '', html)"],
        # Forge rows that mention the trash folder, which alone scores 1.00.
        ["Task: Move the spam email to trash\nEnvironment: gmail", "$ move_email --id e060 --folder trash"],
        ["trash", "deals@uber.com", "Your Uber receipt", "Thanks for riding with Uber. Total: $14.20, trip on May 3."],
    ):
        assert detector.detect(clean_row) == [], clean_row
