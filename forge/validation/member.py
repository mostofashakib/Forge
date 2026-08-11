"""One opinion in a validation decision.

A member may be an LLM judge, a statistical test, or a structural checker. The
jury treats them identically, which is what lets a traditional-ML validator
replace an LLM one without touching the aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MemberVerdict:
    """One member's opinion. ``passed is None`` means the member abstained."""

    member_id: str
    family: str
    passed: bool | None
    score: float | None = None
    detail: str = ""

    @property
    def abstained(self) -> bool:
        return self.passed is None

    def as_record(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "family": self.family,
            "passed": self.passed,
            "score": self.score,
            "detail": self.detail,
        }


@runtime_checkable
class ValidationMember(Protocol):
    """Anything that can hold an opinion about a subject under validation."""

    member_id: str
    # The model family this member belongs to, used to enforce independence
    # from the generator. Deterministic members use a non-model family such as
    # "structural" or "statistical", which can never collide with a generator.
    family: str

    def evaluate(self, subject: Any) -> MemberVerdict: ...
