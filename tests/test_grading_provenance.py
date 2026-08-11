"""Generator/grader independence — who authored the environment vs. who graded the agent.

Structural verification (state assertions, call-order, negative checks) needs no
independence guarantee: it never asks a model for a verdict. LLM judging does,
because a model grading environments authored by its own family is contaminated.
"""
from __future__ import annotations

import pytest

from forge.grading_provenance import (
    GraderContaminationError,
    GradingProvenance,
    model_family,
    require_independent_grader,
    resolve_grading_provenance,
)


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


# ---------------------------------------------------------------------------
# Independence
# ---------------------------------------------------------------------------

def test_structural_only_grading_is_independent_without_any_judge():
    provenance = GradingProvenance(
        generator_models=("claude-haiku-4-5-20251001", "claude-sonnet-4-6"),
        judge_model=None,
        llm_graded=False,
    )
    assert provenance.structural_only
    assert provenance.independent
    assert provenance.judge_family is None


def test_judge_from_a_different_family_is_independent():
    provenance = GradingProvenance(
        generator_models=("claude-sonnet-4-6",),
        judge_model="gpt-4o",
        llm_graded=True,
    )
    assert provenance.independent


def test_judge_sharing_the_generator_family_is_not_independent():
    provenance = GradingProvenance(
        generator_models=("claude-sonnet-4-6",),
        judge_model="claude-sonnet-4-6",
        llm_graded=True,
    )
    assert not provenance.independent


def test_a_different_model_from_the_same_family_is_still_not_independent():
    """False-positive guard: a distinct model id looks like separation but isn't.

    Grading Sonnet-authored environments with Haiku changes the checkpoint, not
    the family. This is the configuration most likely to be mistaken for a
    controlled setup, so it must be rejected.
    """
    provenance = GradingProvenance(
        generator_models=("claude-sonnet-4-6",),
        judge_model="claude-haiku-4-5-20251001",
        llm_graded=True,
    )
    assert not provenance.independent


def test_llm_grading_without_a_configured_judge_is_not_independent():
    provenance = GradingProvenance(
        generator_models=("claude-sonnet-4-6",),
        judge_model=None,
        llm_graded=True,
    )
    assert not provenance.independent


def test_judge_must_differ_from_every_generation_tier():
    """Independence fails if the judge matches *any* model used to generate."""
    provenance = GradingProvenance(
        generator_models=("gpt-4o", "claude-sonnet-4-6"),
        judge_model="gpt-4o-mini",
        llm_graded=True,
    )
    assert not provenance.independent


# ---------------------------------------------------------------------------
# require_independent_grader
# ---------------------------------------------------------------------------

def test_require_independent_grader_accepts_structural_only_grading():
    provenance = GradingProvenance(
        generator_models=("claude-sonnet-4-6",), judge_model=None, llm_graded=False
    )
    require_independent_grader(provenance)  # must not raise


def test_require_independent_grader_rejects_a_same_family_judge():
    provenance = GradingProvenance(
        generator_models=("claude-sonnet-4-6",),
        judge_model="claude-haiku-4-5-20251001",
        llm_graded=True,
    )
    with pytest.raises(GraderContaminationError) as excinfo:
        require_independent_grader(provenance)
    assert "claude" in str(excinfo.value)


def test_require_independent_grader_names_the_variable_that_fixes_it():
    provenance = GradingProvenance(
        generator_models=("claude-sonnet-4-6",),
        judge_model="claude-sonnet-4-6",
        llm_graded=True,
    )
    with pytest.raises(GraderContaminationError) as excinfo:
        require_independent_grader(provenance)
    assert "FORGE_JUDGE_MODEL" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------

def test_resolve_reads_both_generation_tiers_and_the_judge(monkeypatch):
    monkeypatch.setenv("FORGE_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "gpt-4o")

    provenance = resolve_grading_provenance(llm_graded=True)

    assert provenance.generator_models == (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
    )
    assert provenance.judge_model == "gpt-4o"
    assert provenance.independent


def test_resolve_without_a_judge_model_falls_back_to_the_generator(monkeypatch):
    """No FORGE_JUDGE_MODEL means the generator grades itself — never independent."""
    monkeypatch.setenv("FORGE_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")
    monkeypatch.delenv("FORGE_JUDGE_MODEL", raising=False)

    provenance = resolve_grading_provenance(llm_graded=True)

    assert not provenance.independent


def test_resolve_marks_structural_runs_even_when_a_judge_is_configured(monkeypatch):
    """A configured judge that never runs must not be recorded as having graded."""
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "gpt-4o")

    provenance = resolve_grading_provenance(llm_graded=False)

    assert provenance.structural_only
    assert provenance.judge_model is None


def test_provenance_record_is_json_safe_and_carries_the_verdict():
    provenance = GradingProvenance(
        generator_models=("claude-sonnet-4-6",),
        judge_model="gpt-4o",
        llm_graded=True,
    )
    record = provenance.as_record()
    assert record == {
        "generator_models": ["claude-sonnet-4-6"],
        "generator_families": ["claude"],
        "judge_model": "gpt-4o",
        "judge_family": "gpt",
        "llm_graded": True,
        "independent": True,
    }
