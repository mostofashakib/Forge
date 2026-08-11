"""The semantic review gate, as a single judge or an independent quorum.

The reviewer decides whether generated artifacts meet the user's requirements.
Run by the model that wrote those artifacts, that decision is worthless; run by
a panel drawn from other families, a split tells you the decision is contested.
"""
from __future__ import annotations

import pytest

from forge.envgen.agents.semantic_review import PanelResult, SemanticReviewPanel
from forge.grading_provenance import GraderContaminationError
from forge.validation.quorum import QuorumSpec


class _StubClient:
    """Returns a fixed assessment; stands in for one provider's judge."""

    def __init__(self, requirements_met: bool, findings: list[str] | None = None) -> None:
        self._met = requirements_met
        self._findings = findings or []
        self.calls = 0

    def extract(self, system, user, schema):
        self.calls += 1
        return schema(requirements_met=self._met, findings=self._findings)


class _ExplodingClient:
    def extract(self, system, user, schema):
        raise RuntimeError("provider unreachable")


def _panel(clients: dict[str, object], **kwargs) -> SemanticReviewPanel:
    """Build a quorum panel whose members map provider:model -> stub client."""
    specs = tuple(
        QuorumSpec(provider=provider, model=model)
        for provider, model in [("openai", "gpt-4o"), ("gemini", "gemini-2.0-flash"),
                                ("ollama", "llama3.1:8b")][: len(clients)]
    )
    ordered = list(clients.values())
    return SemanticReviewPanel(
        quorum_specs=specs,
        generator_families=("claude",),
        client_factory=lambda spec: ordered[specs.index(spec)],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Single judge (no quorum configured)
# ---------------------------------------------------------------------------

def test_single_judge_returns_its_own_assessment():
    client = _StubClient(requirements_met=True, findings=["minor nit"])
    panel = SemanticReviewPanel(client=client, generator_families=("claude",))

    result = panel.assess("artifacts")

    assert result.requirements_met is True
    assert result.findings == ["minor nit"]
    assert client.calls == 1


def test_single_judge_reporting_unmet_requirements_fails_the_gate():
    panel = SemanticReviewPanel(
        client=_StubClient(requirements_met=False, findings=["missing endpoint"]),
        generator_families=("claude",),
    )

    result = panel.assess("artifacts")

    assert result.requirements_met is False
    assert result.contested is False


# ---------------------------------------------------------------------------
# Quorum
# ---------------------------------------------------------------------------

def test_quorum_consults_every_member():
    clients = {
        "a": _StubClient(True), "b": _StubClient(True), "c": _StubClient(True),
    }
    _panel(clients).assess("artifacts")

    assert all(client.calls == 1 for client in clients.values())


def test_unanimous_agreement_that_requirements_are_met_passes():
    result = _panel({"a": _StubClient(True), "b": _StubClient(True)}).assess("x")

    assert result.requirements_met is True
    assert result.contested is False


def test_unanimous_agreement_that_requirements_are_unmet_fails():
    result = _panel({
        "a": _StubClient(False, ["no reset endpoint"]),
        "b": _StubClient(False, ["reward never fires"]),
    }).assess("x")

    assert result.requirements_met is False
    joined = " ".join(result.findings)
    assert "no reset endpoint" in joined
    assert "reward never fires" in joined
    # Findings are attributed, so a reader can tell which member raised what.
    assert "[gpt-4o]" in joined


def test_a_split_panel_is_contested_not_a_pass():
    """Two-to-one is not agreement; the gate must not report it as approval."""
    result = _panel({
        "a": _StubClient(True), "b": _StubClient(True),
        "c": _StubClient(False, ["state schema is wrong"]),
    }).assess("x")

    assert result.contested is True
    assert result.requirements_met is None


def test_a_contested_panel_keeps_the_dissenting_findings():
    """The dissent is the actionable part — it must survive into the report."""
    result = _panel({
        "a": _StubClient(True), "b": _StubClient(True),
        "c": _StubClient(False, ["state schema is wrong"]),
    }).assess("x")

    assert any("state schema is wrong" in finding for finding in result.findings)


def test_a_member_that_errors_abstains_without_sinking_the_panel():
    result = _panel({
        "a": _StubClient(True), "b": _StubClient(True), "c": _ExplodingClient(),
    }).assess("x")

    assert result.requirements_met is True
    assert result.contested is False


def test_a_panel_where_every_member_errors_is_contested():
    """No evidence is not the same as approval."""
    result = _panel({"a": _ExplodingClient(), "b": _ExplodingClient()}).assess("x")

    assert result.requirements_met is None
    assert result.contested is True


# ---------------------------------------------------------------------------
# Independence
# ---------------------------------------------------------------------------

def test_panel_refuses_a_member_from_the_generating_family():
    with pytest.raises(GraderContaminationError):
        SemanticReviewPanel(
            quorum_specs=(QuorumSpec(provider="anthropic", model="claude-sonnet-4-6"),),
            generator_families=("claude",),
            client_factory=lambda spec: _StubClient(True),
        )


def test_panel_refuses_a_member_of_the_generating_family_under_another_tier():
    """False-positive guard: a different checkpoint is not a different family."""
    with pytest.raises(GraderContaminationError):
        SemanticReviewPanel(
            quorum_specs=(
                QuorumSpec(provider="anthropic", model="claude-haiku-4-5-20251001"),
            ),
            generator_families=("claude",),
            client_factory=lambda spec: _StubClient(True),
        )


def test_panel_requires_either_a_client_or_a_quorum():
    with pytest.raises(ValueError):
        SemanticReviewPanel(generator_families=("claude",))


def test_result_reports_the_member_positions_for_the_record():
    result: PanelResult = _panel({
        "a": _StubClient(True), "b": _StubClient(False, ["bad reward"]),
    }).assess("x")

    record = result.as_record()
    assert record["contested"] is True
    assert len(record["votes"]) == 2
