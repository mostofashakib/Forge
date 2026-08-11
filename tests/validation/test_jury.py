"""Jury aggregation: majority decides, a split is indeterminate.

A jury member may be an LLM or a deterministic checker — the jury does not care
which. What it cares about is that members are independent of the model that
generated what they are judging, and that a contested verdict is reported as
contested rather than resolved by a coin flip.
"""
from __future__ import annotations

import pytest

from forge.validation.jury import Jury, JuryOutcome
from forge.validation.member import MemberVerdict


class _FixedMember:
    """A member with a predetermined verdict. Stands in for an LLM or a checker."""

    def __init__(self, member_id: str, family: str, passed: bool | None, score=None) -> None:
        self.member_id = member_id
        self.family = family
        self._passed = passed
        self._score = score

    def evaluate(self, subject) -> MemberVerdict:
        return MemberVerdict(
            member_id=self.member_id, family=self.family,
            passed=self._passed, score=self._score, detail="fixed",
        )


def _jury(*members, **kwargs) -> Jury:
    return Jury(members=list(members), generator_families=("claude",), **kwargs)


def test_unanimous_pass_decides_pass():
    outcome = _jury(
        _FixedMember("gpt-4o", "gpt", True),
        _FixedMember("gemini-2.0-flash", "gemini", True),
        _FixedMember("llama3.1:8b", "llama", True),
    ).deliberate(subject={})

    assert outcome.decision is True
    assert outcome.indeterminate is False
    assert outcome.agreement == 1.0


def test_unanimous_fail_decides_fail():
    outcome = _jury(
        _FixedMember("gpt-4o", "gpt", False),
        _FixedMember("gemini-2.0-flash", "gemini", False),
    ).deliberate(subject={})

    assert outcome.decision is False
    assert outcome.indeterminate is False


def test_split_verdict_is_indeterminate_not_a_majority_win():
    """The whole point: a 2-1 split must not be reported as a clean verdict."""
    outcome = _jury(
        _FixedMember("gpt-4o", "gpt", True),
        _FixedMember("gemini-2.0-flash", "gemini", True),
        _FixedMember("llama3.1:8b", "llama", False),
    ).deliberate(subject={})

    assert outcome.indeterminate is True
    assert outcome.decision is None
    assert outcome.agreement == pytest.approx(2 / 3)


def test_split_becomes_decisive_when_the_threshold_is_lowered():
    outcome = _jury(
        _FixedMember("gpt-4o", "gpt", True),
        _FixedMember("gemini-2.0-flash", "gemini", True),
        _FixedMember("llama3.1:8b", "llama", False),
        agreement_threshold=0.67,
    ).deliberate(subject={})

    assert outcome.decision is True
    assert outcome.indeterminate is False


def test_abstaining_member_is_excluded_from_the_denominator():
    """An abstention is not a vote against — it must not dilute agreement."""
    outcome = _jury(
        _FixedMember("gpt-4o", "gpt", True),
        _FixedMember("gemini-2.0-flash", "gemini", True),
        _FixedMember("llama3.1:8b", "llama", None),
    ).deliberate(subject={})

    assert outcome.agreement == 1.0
    assert outcome.decision is True


def test_jury_where_every_member_abstains_is_indeterminate():
    outcome = _jury(
        _FixedMember("gpt-4o", "gpt", None),
        _FixedMember("gemini-2.0-flash", "gemini", None),
    ).deliberate(subject={})

    assert outcome.indeterminate is True
    assert outcome.decision is None


def test_every_member_verdict_is_retained_for_the_record():
    outcome = _jury(
        _FixedMember("gpt-4o", "gpt", True),
        _FixedMember("gemini-2.0-flash", "gemini", False),
    ).deliberate(subject={})

    assert [vote.member_id for vote in outcome.votes] == ["gpt-4o", "gemini-2.0-flash"]


# ---------------------------------------------------------------------------
# Independence
# ---------------------------------------------------------------------------

def test_jury_refuses_a_member_from_the_generating_family():
    from forge.grading_provenance import GraderContaminationError

    with pytest.raises(GraderContaminationError):
        _jury(
            _FixedMember("gpt-4o", "gpt", True),
            _FixedMember("claude-sonnet-4-6", "claude", True),
        )


def test_jury_refuses_a_look_independent_member_of_the_same_family():
    """False-positive guard: a different model id is not a different family."""
    from forge.grading_provenance import GraderContaminationError

    with pytest.raises(GraderContaminationError) as excinfo:
        Jury(
            members=[_FixedMember("claude-haiku-4-5-20251001", "claude", True)],
            generator_families=("claude",),
        )
    assert "claude" in str(excinfo.value)


def test_jury_refuses_to_convene_with_no_members():
    with pytest.raises(ValueError):
        Jury(members=[], generator_families=("claude",))


def test_jury_rejects_an_agreement_threshold_above_one():
    with pytest.raises(ValueError):
        _jury(_FixedMember("gpt-4o", "gpt", True), agreement_threshold=1.5)


def test_jury_rejects_an_agreement_threshold_at_or_below_half():
    """Below a majority the 'agreement' threshold would decide nothing."""
    with pytest.raises(ValueError):
        _jury(_FixedMember("gpt-4o", "gpt", True), agreement_threshold=0.5)


# ---------------------------------------------------------------------------
# Deterministic members
# ---------------------------------------------------------------------------

def test_a_deterministic_member_votes_like_any_other():
    """A structural checker and an LLM judge are the same thing to the jury."""

    class _StructuralMember:
        member_id = "layered-verifier"
        family = "structural"

        def evaluate(self, subject) -> MemberVerdict:
            return MemberVerdict(
                member_id=self.member_id, family=self.family,
                passed=subject["milestones_met"], score=None, detail="structural",
            )

    outcome = _jury(
        _StructuralMember(),
        _FixedMember("gpt-4o", "gpt", True),
    ).deliberate(subject={"milestones_met": True})

    assert outcome.decision is True
    assert "layered-verifier" in [vote.member_id for vote in outcome.votes]


def test_outcome_record_is_json_safe():
    outcome: JuryOutcome = _jury(
        _FixedMember("gpt-4o", "gpt", True),
        _FixedMember("gemini-2.0-flash", "gemini", False),
    ).deliberate(subject={})

    record = outcome.as_record()
    assert record["decision"] is None
    assert record["indeterminate"] is True
    assert record["votes"][0]["member_id"] == "gpt-4o"
    assert record["votes"][1]["passed"] is False


def test_a_threshold_written_as_two_thirds_accepts_a_genuine_two_thirds_majority():
    """Guard the rounding trap: 2/3 = 0.666… must not fall under a 0.67 threshold."""
    outcome = _jury(
        _FixedMember("gpt-4o", "gpt", True),
        _FixedMember("gemini-2.0-flash", "gemini", True),
        _FixedMember("llama3.1:8b", "llama", False),
        agreement_threshold=0.67,
    ).deliberate(subject={})

    assert outcome.decision is True


def test_the_tolerance_does_not_let_a_bare_majority_pass_a_unanimity_threshold():
    """False-positive guard: tolerance must not quietly weaken unanimity."""
    outcome = _jury(
        _FixedMember("gpt-4o", "gpt", True),
        _FixedMember("gemini-2.0-flash", "gemini", True),
        _FixedMember("llama3.1:8b", "llama", False),
        agreement_threshold=1.0,
    ).deliberate(subject={})

    assert outcome.decision is None
