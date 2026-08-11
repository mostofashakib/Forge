"""The semantic half of the generated-environment review gate.

Separated from :class:`ReviewerAgent` because it is the one part of the gate
that asks a model for a judgement, and therefore the one part that has to be
independent of the model that wrote the artifacts. Running it as a quorum turns
a single opinion into a measurable level of agreement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from forge.validation.jury import DEFAULT_AGREEMENT_THRESHOLD, Jury, JuryOutcome
from forge.validation.quorum import QuorumSpec, quorum_members

_REVIEW_SYSTEM = (
    "You are the final reviewer for a generated reinforcement-learning environment. "
    "Compare the user's request and structured domain requirements to the supplied artifacts. "
    "Check functional coverage, UI-to-API action coverage, RL state/reward suitability, and "
    "clear code responsibilities. Report only concrete unmet requirements or code-quality "
    "problems; do not reject for subjective style preferences."
)


class SemanticReviewPrompts:
    SYSTEM = _REVIEW_SYSTEM


class RequirementAssessment(BaseModel):
    requirements_met: bool
    findings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class PanelResult:
    """The panel's position on whether requirements are met.

    ``requirements_met is None`` means the panel is contested: the members did
    not agree, so the gate has no verdict to report.
    """

    requirements_met: bool | None
    findings: list[str]
    outcome: JuryOutcome | None = None

    @property
    def contested(self) -> bool:
        return self.requirements_met is None

    def as_record(self) -> dict[str, Any]:
        return {
            "requirements_met": self.requirements_met,
            "contested": self.contested,
            "findings": list(self.findings),
            "votes": (
                [vote.as_record() for vote in self.outcome.votes]
                if self.outcome is not None else []
            ),
        }


def _assess(client: Any, semantic_input: str) -> tuple[bool, str]:
    assessment: RequirementAssessment = client.extract(
        system=SemanticReviewPrompts.SYSTEM,
        user=semantic_input,
        schema=RequirementAssessment,
    )
    return assessment.requirements_met, "; ".join(assessment.findings)


class SemanticReviewPanel:
    """Runs the semantic requirements review as a single judge or a quorum."""

    def __init__(
        self,
        *,
        generator_families: tuple[str, ...],
        client: Any | None = None,
        quorum_specs: Sequence[QuorumSpec] | None = None,
        agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
        client_factory: Callable[[QuorumSpec], Any] | None = None,
    ) -> None:
        if client is None and not quorum_specs:
            raise ValueError(
                "semantic review needs either a judge client or a quorum; "
                "pass client= or set FORGE_QUORUM_MODELS"
            )
        self._client = client
        self._jury: Jury | None = None
        if quorum_specs:
            factory_kwargs = (
                {"client_factory": client_factory} if client_factory is not None else {}
            )
            members = quorum_members(
                quorum_specs,
                generator_families=generator_families,
                evaluate=_assess,
                **factory_kwargs,
            )
            self._jury = Jury(
                members=members,
                generator_families=generator_families,
                agreement_threshold=agreement_threshold,
            )

    def assess(self, semantic_input: str) -> PanelResult:
        if self._jury is None:
            met, detail = _assess(self._client, semantic_input)
            return PanelResult(
                requirements_met=met,
                findings=[part for part in detail.split("; ") if part],
            )

        outcome = self._jury.deliberate(semantic_input)
        # Keep every member's findings, including the dissent: when the panel is
        # contested the dissenting finding is the actionable part, and it is
        # exactly what a majority vote would have discarded.
        findings = [
            f"[{vote.member_id}] {vote.detail}"
            for vote in outcome.votes
            if vote.detail
        ]
        return PanelResult(
            requirements_met=outcome.decision,
            findings=findings,
            outcome=outcome,
        )
