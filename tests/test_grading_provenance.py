"""Model families collapse tiers of one vendor so provenance can be compared."""
from __future__ import annotations

import pytest

from forge.grading_provenance import model_family


# ---------------------------------------------------------------------------
# model_family
# ---------------------------------------------------------------------------

def test_model_family_collapses_tiers_of_the_same_model():
    assert model_family("claude-haiku-4-5-20251001") == "claude"
    assert model_family("claude-sonnet-4-6") == "claude"


def test_model_family_reads_the_vendor_from_a_namespaced_id():
    assert model_family("Qwen/Qwen2.5-3B-Instruct") == "qwen"


def test_model_family_separates_distinct_vendors():
    assert model_family("claude-sonnet-4-6") != model_family("gemma4:26b")
    assert model_family("gpt-4o") != model_family("llama-3.1-8b")


def test_model_family_rejects_an_id_with_no_alphabetic_prefix():
    with pytest.raises(ValueError):
        model_family("4-5-20251001")


def test_model_family_rejects_a_blank_id():
    with pytest.raises(ValueError):
        model_family("   ")
