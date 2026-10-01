"""Shared pieces of the SFT and GRPO backends, testable without torch."""
from __future__ import annotations

from pathlib import Path

from forge.training._backends import _load_tokenizer, _tokenize_completion, _training_arguments


class _Tokenizer:
    """Maps each whitespace word to its length; BOS is 0 when special tokens are added."""

    def __init__(self, eos_token_id=99, pad_token_id=None) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token = "</s>"
        self.pad_token = None

    def __call__(self, text, add_special_tokens):
        ids = [len(word) for word in text.split()]
        return {"input_ids": [0, *ids] if add_special_tokens else ids}


def test_only_completion_tokens_carry_labels_and_eos_ends_them():
    row = _tokenize_completion(_Tokenizer(), "fix it", "run tests")

    assert row["input_ids"] == [0, 3, 2, 3, 5, 99]
    assert row["labels"] == [-100, -100, -100, 3, 5, 99]


def test_a_tokenizer_without_eos_appends_nothing():
    row = _tokenize_completion(_Tokenizer(eos_token_id=None), "go", "now")

    assert row["input_ids"] == [0, 2, 3]
    assert row["labels"] == [-100, -100, 3]


def test_an_empty_completion_trains_only_on_eos():
    row = _tokenize_completion(_Tokenizer(), "go", "")

    assert row["labels"] == [-100, -100, 99]


def test_a_tokenizer_without_padding_pads_with_eos():
    class Auto:
        @staticmethod
        def from_pretrained(name):
            assert name == "base"
            return _Tokenizer(pad_token_id=None)

    assert _load_tokenizer(Auto, "base").pad_token == "</s>"


def test_a_tokenizer_with_padding_keeps_it():
    class Auto:
        @staticmethod
        def from_pretrained(name):
            tokenizer = _Tokenizer(pad_token_id=1)
            tokenizer.pad_token = "<pad>"
            return tokenizer

    assert _load_tokenizer(Auto, "base").pad_token == "<pad>"


def _recorded_arguments(num_rows: int) -> dict:
    return _training_arguments(lambda **kwargs: kwargs, Path("/out"), max_steps=40, num_rows=num_rows)


def test_training_arguments_write_state_under_the_output_dir():
    arguments = _recorded_arguments(3)

    assert arguments["output_dir"] == "/out/trainer_state"
    assert arguments["max_steps"] == 40
    assert arguments["gradient_accumulation_steps"] == 3
    assert arguments["save_strategy"] == "no"
    assert arguments["remove_unused_columns"] is False


def test_gradient_accumulation_is_capped_at_eight_and_never_zero():
    assert _recorded_arguments(50)["gradient_accumulation_steps"] == 8
    assert _recorded_arguments(0)["gradient_accumulation_steps"] == 1
