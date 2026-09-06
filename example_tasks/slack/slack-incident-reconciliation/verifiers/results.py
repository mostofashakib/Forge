"""The verdict shapes, built so a score can always be explained.

A number on its own is not a grade: the useful artefact is the line-by-line
account of what was checked, what passed, and what each layer contributed. So
every verifier returns its individual checks, the breakdown keeps each layer's
weight and contribution separately from the penalties applied afterwards, and
the total is derived from those parts rather than asserted alongside them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The deterministic grading paths.
LAYERS = ("final_state", "milestones", "trajectory", "negative")


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """One assertion inside one verifier."""

    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "weight": self.weight,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class Penalty:
    reason: str
    amount: float
    source: str = "auditor"

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "amount": round(self.amount, 6), "source": self.source}


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """What one verifier concluded, and why."""

    verifier: str
    layer: str
    score: float
    checks: tuple[CheckOutcome, ...] = ()
    #: A failure here sinks the episode whatever the arithmetic says. Reserved
    #: for the two things partial credit must never launder: a forbidden side
    #: effect, and milestones reached out of order.
    fatal_on_failure: bool = False
    #: False when a configured layer could not run at all. Distinct from
    #: scoring zero, which means it ran and found nothing.
    available: bool = True
    #: Failed checks that end the episode outright rather than costing it
    #: something. Reserved for acts that are not degrees of doing the task
    #: badly: leaving the environment, or tampering with what grades it.
    disqualifying_checks: tuple[str, ...] = ()
    #: Charges this verifier found, for faults that cost reward without ending
    #: the episode. Reported here for the same reason vetoes are: the verifier
    #: that found the fault decides what it is worth, and the engine applies
    #: what it reports rather than re-deriving it by a second route.
    penalties: tuple[Penalty, ...] = ()

    @property
    def passed(self) -> bool:
        return self.available and all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "layer": self.layer,
            "score": round(self.score, 6),
            "passed": self.passed,
            "available": self.available,
            "fatal_on_failure": self.fatal_on_failure,
            "failed_checks": list(self.failed_checks),
            "disqualifying_checks": list(self.disqualifying_checks),
            "penalties": [penalty.as_dict() for penalty in self.penalties],
            "checks": [check.as_dict() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class LayerScore:
    layer: str
    weight: float
    score: float
    available: bool = True

    @property
    def contribution(self) -> float:
        return self.weight * self.score if self.available else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "weight": round(self.weight, 6),
            "score": round(self.score, 6),
            "available": self.available,
            "contribution": round(self.contribution, 6),
        }


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Every term that produced the number, in the order it was applied."""

    layers: tuple[LayerScore, ...] = ()
    penalties: tuple[Penalty, ...] = ()
    #: What the layers came to before penalties, for reading a run at a glance.
    base: float = 0.0

    @property
    def penalty_total(self) -> float:
        return sum(penalty.amount for penalty in self.penalties)

    @property
    def total(self) -> float:
        return max(0.0, min(1.0, self.base - self.penalty_total))

    def as_dict(self) -> dict[str, Any]:
        return {
            "layers": [layer.as_dict() for layer in self.layers],
            "penalties": [penalty.as_dict() for penalty in self.penalties],
            "base": round(self.base, 6),
            "penalty_total": round(self.penalty_total, 6),
            "total": round(self.total, 6),
        }


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    """The final verdict: pass or fail, the reward, and the whole audit trail."""

    preset: str
    passed: bool
    reward: float
    breakdown: RewardBreakdown
    results: tuple[VerifierResult, ...] = ()
    audit: dict[str, Any] | None = None
    #: False when grading itself could not be carried out -- a missing state
    #: export, for example. A broken grader must never reach an experiment as
    #: a score of zero.
    valid: bool = True
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "passed": self.passed,
            "reward": round(self.reward, 6),
            "valid": self.valid,
            "notes": list(self.notes),
            "breakdown": self.breakdown.as_dict(),
            "results": [result.as_dict() for result in self.results],
            "audit": self.audit,
        }

    def summary(self) -> str:
        lines = [
            f"preset={self.preset}  passed={self.passed}  "
            f"reward={self.reward:.4f}  valid={self.valid}"
        ]
        for layer in self.breakdown.layers:
            state = "" if layer.available else "  (unavailable)"
            lines.append(
                f"  {layer.layer:<12} weight={layer.weight:.2f} "
                f"score={layer.score:.3f} -> {layer.contribution:.4f}{state}"
            )
        for penalty in self.breakdown.penalties:
            lines.append(f"  penalty -{penalty.amount:.4f}  {penalty.reason}")
        for result in self.results:
            if result.failed_checks:
                lines.append(f"  {result.layer}/{result.verifier} failed: "
                             f"{', '.join(result.failed_checks)}")
        return "\n".join(lines)
