"""Aggregate independent opinions into one validation decision.

Majority decides. A verdict that the members do not agree on is reported as
*indeterminate* rather than resolved by a narrow majority: a contested decision
is a fact about the instrument, and hiding it behind a clean pass/fail is how a
weak benchmark comes to look strong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from forge.grading_provenance import GraderContaminationError
from forge.validation.member import MemberVerdict, ValidationMember

# Unanimity among voting members. Lower it to ~0.67 to accept 2-1 majorities on
# a three-member jury.
DEFAULT_AGREEMENT_THRESHOLD = 1.0

# Thresholds are user-facing config, and a person writing 0.67 means "two
# thirds". Without a tolerance, 2/3 = 0.666… would fall just under 0.67 and
# abstain — the opposite of what was configured.
_AGREEMENT_TOLERANCE = 0.01


@dataclass(frozen=True)
class JuryOutcome:
    """The jury's decision plus every opinion that produced it."""

    decision: bool | None
    votes: tuple[MemberVerdict, ...]
    agreement: float
    reason: str = ""

    @property
    def indeterminate(self) -> bool:
        return self.decision is None

    def as_record(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "indeterminate": self.indeterminate,
            "agreement": self.agreement,
            "reason": self.reason,
            "votes": [vote.as_record() for vote in self.votes],
        }


@dataclass
class Jury:
    """A panel of independent members that votes on one kind of decision."""

    members: Sequence[ValidationMember]
    generator_families: tuple[str, ...]
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD
    _checked: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("a jury needs at least one member")
        if not 0.5 < self.agreement_threshold <= 1.0:
            raise ValueError(
                "agreement_threshold must be above 0.5 and at most 1.0; "
                f"got {self.agreement_threshold}"
            )
        self._reject_generator_families()

    def _reject_generator_families(self) -> None:
        """A member from the generating family is marking its own homework."""
        generators = set(self.generator_families)
        contaminated = [
            member for member in self.members if member.family in generators
        ]
        if contaminated:
            names = [member.member_id for member in contaminated]
            raise GraderContaminationError(
                f"jury members {names} share family "
                f"{sorted({m.family for m in contaminated})} with the generating "
                "models; a quorum drawn from the generating family shares its "
                "blind spots and is not independent evidence"
            )

    def deliberate(self, subject: Any) -> JuryOutcome:
        votes = tuple(member.evaluate(subject) for member in self.members)
        voting = [vote for vote in votes if not vote.abstained]

        if not voting:
            return JuryOutcome(
                decision=None, votes=votes, agreement=0.0,
                reason="every member abstained",
            )

        passes = sum(1 for vote in voting if vote.passed)
        majority = passes * 2 > len(voting)
        # Agreement is measured only among members that actually voted, so an
        # abstention never counts as a vote against.
        agreeing = passes if majority else len(voting) - passes
        agreement = agreeing / len(voting)

        if agreement < self.agreement_threshold - _AGREEMENT_TOLERANCE:
            return JuryOutcome(
                decision=None, votes=votes, agreement=agreement,
                reason=(
                    f"members split {passes}/{len(voting)}; agreement "
                    f"{agreement:.2f} is below the {self.agreement_threshold:.2f} "
                    "threshold"
                ),
            )
        return JuryOutcome(
            decision=majority, votes=votes, agreement=agreement,
            reason=f"{agreeing}/{len(voting)} members agreed",
        )
