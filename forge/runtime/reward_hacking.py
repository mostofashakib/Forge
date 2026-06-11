from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from forge.runtime.verification import VerificationResult

# An LLM audit client takes (state, trajectory, task, verification) and
# returns (flagged, detail).
AuditClient = Callable[[dict, object, dict, VerificationResult], tuple[bool, str]]


@dataclass
class AuditFinding:
    code: str
    severity: str  # "high" | "medium" | "low"
    detail: str


@dataclass
class AuditReport:
    flagged: bool
    findings: list[AuditFinding] = field(default_factory=list)


class RewardHackingAuditor:
    """Separate audit agent that reviews a verifier's verdict for reward hacking.

    Programmatic milestone verifiers say *whether* the task passed; this agent
    asks whether the pass was earned. It cross-checks the trajectory against
    the required milestones, flags passes that look too cheap, spots redundant
    call patterns, and can delegate to an LLM client for a deeper audit.
    """

    def __init__(
        self,
        required_milestones: list[str] | None = None,
        min_steps: int = 1,
        max_action_repeats: int = 3,
        llm_client: AuditClient | None = None,
    ) -> None:
        self._required_milestones = required_milestones or []
        self._min_steps = min_steps
        self._max_action_repeats = max_action_repeats
        self._llm_client = llm_client

    @classmethod
    def for_verifier(cls, verifier, **kwargs) -> "RewardHackingAuditor":
        """Build an auditor that inherits the milestone list of a LayeredVerifier."""
        return cls(required_milestones=list(verifier.required_milestones), **kwargs)

    def audit(
        self, state: dict, trajectory, task: dict, verification: VerificationResult
    ) -> AuditReport:
        findings: list[AuditFinding] = []

        if verification.passed:
            event_types = {e.get("type") for e in trajectory.events}
            skipped = [m for m in self._required_milestones if m not in event_types]
            if skipped:
                findings.append(AuditFinding(
                    code="milestones_skipped",
                    severity="high",
                    detail=f"verifier passed but milestones never reached: {skipped}",
                ))
            if trajectory.step_count < self._min_steps:
                findings.append(AuditFinding(
                    code="suspiciously_short",
                    severity="medium",
                    detail=(
                        f"verifier passed after {trajectory.step_count} step(s); "
                        f"legitimate completion needs at least {self._min_steps}"
                    ),
                ))

        repeats = self._consecutive_repeats(trajectory)
        if repeats:
            findings.append(AuditFinding(
                code="redundant_actions",
                severity="medium",
                detail=f"identical action repeated consecutively: {repeats}",
            ))

        if self._llm_client is not None:
            flagged, detail = self._llm_client(state, trajectory, task, verification)
            if flagged:
                findings.append(AuditFinding(code="llm_audit", severity="high", detail=detail))

        return AuditReport(flagged=bool(findings), findings=findings)

    def _consecutive_repeats(self, trajectory) -> list[dict]:
        offending: list[dict] = []
        run_action: dict | None = None
        run_length = 0
        for step in trajectory.steps:
            if step.action == run_action:
                run_length += 1
            else:
                run_action = step.action
                run_length = 1
            if run_length == self._max_action_repeats + 1:
                offending.append(run_action)
        return offending
