"""Building a quorum from configuration, and refusing a contaminated one."""
from __future__ import annotations

import pytest

from forge.grading_provenance import GraderContaminationError
from forge.validation.quorum import QuorumSpec, parse_quorum_models, quorum_members


# ---------------------------------------------------------------------------
# Parsing FORGE_QUORUM_MODELS
# ---------------------------------------------------------------------------

def test_parses_a_provider_model_list():
    specs = parse_quorum_models("openai:gpt-4o, gemini:gemini-2.0-flash")

    assert specs == (
        QuorumSpec(provider="openai", model="gpt-4o"),
        QuorumSpec(provider="gemini", model="gemini-2.0-flash"),
    )


def test_an_empty_setting_yields_no_members():
    assert parse_quorum_models("") == ()
    assert parse_quorum_models("   ") == ()


def test_rejects_an_entry_without_a_provider():
    with pytest.raises(ValueError, match="provider:model"):
        parse_quorum_models("gpt-4o")


def test_rejects_an_entry_with_a_blank_model():
    with pytest.raises(ValueError):
        parse_quorum_models("openai:")


def test_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        parse_quorum_models("not-a-provider:some-model")


def test_rejects_a_duplicate_member():
    """The same model twice is one opinion counted twice, not two opinions."""
    with pytest.raises(ValueError, match="duplicate"):
        parse_quorum_models("openai:gpt-4o,openai:gpt-4o")


def test_rejects_two_members_of_the_same_family():
    """False-positive guard: two OpenAI models look like two members, but one family."""
    with pytest.raises(ValueError, match="family"):
        parse_quorum_models("openai:gpt-4o,openai:gpt-4o-mini")


# ---------------------------------------------------------------------------
# Member construction
# ---------------------------------------------------------------------------

def test_members_carry_their_model_family():
    members = quorum_members(
        parse_quorum_models("openai:gpt-4o,gemini:gemini-2.0-flash"),
        generator_families=("claude",),
        evaluate=lambda client, subject: True,
    )

    assert [member.family for member in members] == ["gpt", "gemini"]


def test_refuses_a_member_from_the_generating_family():
    with pytest.raises(GraderContaminationError):
        quorum_members(
            parse_quorum_models("anthropic:claude-sonnet-4-6"),
            generator_families=("claude",),
            evaluate=lambda client, subject: True,
        )


def test_refuses_an_empty_quorum_when_one_was_requested():
    with pytest.raises(ValueError):
        quorum_members((), generator_families=("claude",), evaluate=lambda c, s: True)


def test_a_member_that_raises_abstains_rather_than_failing_the_run():
    """One unreachable provider must not take down an entire evaluation."""

    def exploding(client, subject):
        raise RuntimeError("provider unreachable")

    members = quorum_members(
        parse_quorum_models("openai:gpt-4o"),
        generator_families=("claude",),
        evaluate=exploding,
        client_factory=lambda spec: object(),
    )
    verdict = members[0].evaluate({})

    assert verdict.abstained
    assert "unreachable" in verdict.detail


def test_a_member_that_returns_a_verdict_does_not_abstain():
    members = quorum_members(
        parse_quorum_models("openai:gpt-4o"),
        generator_families=("claude",),
        evaluate=lambda client, subject: False,
        client_factory=lambda spec: object(),
    )
    verdict = members[0].evaluate({})

    assert verdict.abstained is False
    assert verdict.passed is False


def test_a_member_returning_passed_and_detail_records_both():
    """A (passed, detail) return must be unpacked — a truthy tuple is not a pass."""
    members = quorum_members(
        parse_quorum_models("openai:gpt-4o"),
        generator_families=("claude",),
        evaluate=lambda client, subject: (False, "reward never fires"),
        client_factory=lambda spec: object(),
    )
    verdict = members[0].evaluate({})

    assert verdict.passed is False
    assert verdict.detail == "reward never fires"


def test_a_member_returning_a_bare_bool_still_works():
    members = quorum_members(
        parse_quorum_models("openai:gpt-4o"),
        generator_families=("claude",),
        evaluate=lambda client, subject: True,
        client_factory=lambda spec: object(),
    )

    assert members[0].evaluate({}).passed is True
