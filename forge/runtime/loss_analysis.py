"""Per-run failure-mode (loss) analysis (TASKS.md #4).

Classifies why an agent run went wrong into a fixed seven-mode taxonomy, using
the structured trace from :class:`~forge.runtime.agent_logger.AgentRunLogger` and
the composed verifier's :class:`~forge.runtime.verification.VerificationResult`.
Reward hacking reuses the verdict of
:class:`~forge.runtime.reward_hacking.RewardHackingAuditor`; cross-run
aggregation reuses the :class:`~forge.runtime.clustering.FailureCluster`
primitive.

The analyzer is deliberately deterministic and heuristic — it reads signals the
runtime already records (tool results, the final response, the verifier's failed
tiers, the auditor's findings) rather than calling an LLM. Every mode except
reward hacking is only considered when the run actually failed verification, so a
clean, correct run yields an empty report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from forge.runtime.agent_logger import ACTION, LLM_CALL, AgentRunLogger
from forge.runtime.clustering import FailureCluster
from forge.runtime.reward_hacking import AuditReport
from forge.runtime.verification import VerificationResult


class FailureMode(str, Enum):
    """The fixed taxonomy of agent-run failure modes."""

    INSTRUCTION_FOLLOWING = "instruction_following"
    HALLUCINATION = "hallucination"
    TOOL_SEQUENCING = "tool_sequencing"
    EARLY_STOPPING = "early_stopping"
    CONTEXT_LOSS = "context_loss"
    REWARD_HACKING = "reward_hacking"
    SURFACE_OVERFITTING = "surface_overfitting"


@dataclass
class FailureSignal:
    """One classified failure mode with the evidence that triggered it."""

    mode: FailureMode
    confidence: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "confidence": self.confidence, "evidence": self.evidence}


@dataclass
class RunLossReport:
    """Structured per-run loss analysis, keyed to the run id."""

    run_id: str
    passed: bool
    signals: list[FailureSignal] = field(default_factory=list)

    @property
    def modes(self) -> list[FailureMode]:
        return [s.mode for s in self.signals]

    @property
    def mode_names(self) -> list[str]:
        return [s.mode.value for s in self.signals]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "signals": [s.to_dict() for s in self.signals],
        }


# Function/meta words (length >= 4) that must not count as hallucinated entities.
_STOPWORDS = frozenset({
    "that", "this", "with", "from", "have", "been", "will", "your", "their",
    "there", "which", "what", "when", "where", "answer", "because", "about",
    "into", "then", "than", "them", "they", "these", "those", "would", "could",
    "should", "were", "done", "here", "none", "null", "true", "false", "value",
    "result", "response", "cannot", "unable", "unknown", "match", "matched",
})

_HEDGE_PHRASES = (
    "not sure", "unsure", "cannot", "can't", "don't know", "do not know",
    "unable", "unknown", "not certain", "i'm not", "no idea",
)


def _text(obj: Any) -> str:
    """Concatenate the *string values* of an object (ignoring dict keys).

    Keys like ``owner`` or ``results`` are structural, not content — matching on
    them would create phantom entities, so only leaf string values contribute.
    """
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif value is not None:
            parts.append(str(value))

    walk(obj)
    return " ".join(parts)


def _words(text: str) -> set[str]:
    token = []
    out: set[str] = set()
    for ch in text.lower():
        if ch.isalnum():
            token.append(ch)
        elif token:
            word = "".join(token)
            if len(word) >= 4 and not word.isdigit():
                out.add(word)
            token = []
    if token:
        word = "".join(token)
        if len(word) >= 4 and not word.isdigit():
            out.add(word)
    return out


def _result_is_empty(result: Any) -> bool:
    if not result:
        return True
    if isinstance(result, dict):
        for key in ("results", "matches", "items", "records", "rows", "data", "hits"):
            if key in result and not result[key]:
                return True
        if result.get("count") == 0:
            return True
    if isinstance(result, (list, tuple)):
        return len(result) == 0
    return False


def _string_values(obj: Any) -> list[str]:
    out: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(obj)
    return out


class LossAnalyzer:
    """Classifies an agent run's failure into the :class:`FailureMode` taxonomy."""

    # Base confidences per mode; reward hacking is derived from finding severity.
    _CONFIDENCE = {
        FailureMode.INSTRUCTION_FOLLOWING: 0.85,
        FailureMode.TOOL_SEQUENCING: 0.85,
        FailureMode.CONTEXT_LOSS: 0.8,
        FailureMode.EARLY_STOPPING: 0.7,
        FailureMode.HALLUCINATION: 0.7,
        FailureMode.SURFACE_OVERFITTING: 0.65,
    }
    _SEVERITY_CONFIDENCE = {"high": 0.9, "medium": 0.7, "low": 0.5}

    def __init__(self, *, early_stopping_max_steps: int = 2, min_overfit_chars: int = 12) -> None:
        self._early_stopping_max_steps = early_stopping_max_steps
        self._min_overfit_chars = min_overfit_chars

    def analyze(
        self,
        logger: AgentRunLogger,
        verification: VerificationResult,
        task: dict | None = None,
        audit_report: AuditReport | None = None,
    ) -> RunLossReport:
        task = task or {}
        entries = list(logger.entries)
        actions = [e for e in entries if e.kind == ACTION]
        llm_calls = [e for e in entries if e.kind == LLM_CALL]
        instruction = str(task.get("instruction") or task.get("description") or "")
        expected = task.get("expected_answer")

        final_answer = self._final_answer(llm_calls)
        tool_results = [e.payload.get("result") for e in actions]

        signals: dict[FailureMode, FailureSignal] = {}

        def emit(mode: FailureMode, evidence: str, confidence: float | None = None) -> None:
            signals[mode] = FailureSignal(
                mode=mode,
                confidence=confidence if confidence is not None else self._CONFIDENCE[mode],
                evidence=evidence,
            )

        # Reward hacking is independent of the pass/fail verdict — a cheap pass is
        # exactly what it catches.
        rh = self._reward_hacking(audit_report)
        if rh is not None:
            signals[FailureMode.REWARD_HACKING] = rh

        if not verification.passed:
            self._instruction_following(verification, tool_results, emit)
            self._tool_sequencing(verification, emit)
            self._hallucination(final_answer, tool_results, instruction, emit)
            self._early_stopping(actions, final_answer, emit)
            self._context_loss(actions, final_answer, expected, emit)
            self._surface_overfitting(actions, instruction, emit)

        ordered = [signals[m] for m in FailureMode if m in signals]
        return RunLossReport(run_id=logger.run_id, passed=verification.passed, signals=ordered)

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _reward_hacking(self, audit_report: AuditReport | None) -> FailureSignal | None:
        if audit_report is None or not audit_report.flagged:
            return None
        detail = "; ".join(f.detail for f in audit_report.findings)
        severity = max(
            (f.severity for f in audit_report.findings),
            key=lambda s: self._SEVERITY_CONFIDENCE.get(s, 0.0),
            default="medium",
        )
        return FailureSignal(
            mode=FailureMode.REWARD_HACKING,
            confidence=self._SEVERITY_CONFIDENCE.get(severity, 0.7),
            evidence=detail or "reward-hacking audit flagged the run",
        )

    def _instruction_following(self, verification, tool_results, emit) -> None:
        # A stated constraint was violated (a forbidden action/event/state) while
        # the informing data was available in a tool result.
        if not any(r for r in tool_results):
            return
        for check in verification.checks:
            if check.passed:
                continue
            if "forbidden" in check.name and "action_sequence" not in check.name:
                emit(
                    FailureMode.INSTRUCTION_FOLLOWING,
                    f"violated a stated constraint despite available data: "
                    f"{check.name} — {check.evidence}",
                )
                return

    def _tool_sequencing(self, verification, emit) -> None:
        for check in verification.checks:
            if check.passed:
                continue
            evidence = (check.evidence or "").lower()
            if "action_sequence" in check.name or "out of order" in evidence or "not in order" in evidence:
                emit(
                    FailureMode.TOOL_SEQUENCING,
                    f"tools called in the wrong order: {check.name} — {check.evidence}",
                )
                return

    def _hallucination(self, final_answer, tool_results, instruction, emit) -> None:
        if final_answer is None:
            return
        answer_words = _words(_text(final_answer))
        grounded = _words(" ".join(_text(r) for r in tool_results if r)) | _words(instruction)
        ungrounded = sorted(w for w in answer_words if w not in grounded and w not in _STOPWORDS)
        if ungrounded:
            emit(
                FailureMode.HALLUCINATION,
                f"final answer references entities absent from every tool result: {ungrounded}",
            )

    def _early_stopping(self, actions, final_answer, emit) -> None:
        if final_answer is None or len(actions) > self._early_stopping_max_steps:
            return
        lowered = _text(final_answer).lower()
        if any(phrase in lowered for phrase in _HEDGE_PHRASES):
            return  # hedged, not a confident wrong stop
        emit(
            FailureMode.EARLY_STOPPING,
            f"stopped after {len(actions)} step(s) with a confident but wrong answer: "
            f"{_text(final_answer)!r}",
        )

    def _context_loss(self, actions, final_answer, expected, emit) -> None:
        if not expected or len(actions) < 2:
            return
        needle = str(expected).lower()
        final_text = _text(final_answer).lower() if final_answer is not None else ""
        if needle in final_text:
            return  # the fact was carried through — no loss
        last_action_step = actions[-1].step
        for entry in actions:
            if entry.step is not None and last_action_step is not None and entry.step >= last_action_step:
                continue
            if needle in _text(entry.payload.get("result")).lower():
                emit(
                    FailureMode.CONTEXT_LOSS,
                    f"correct fact {expected!r} appeared in an early tool result "
                    f"(step {entry.step}) but was dropped from the final answer",
                )
                return

    def _surface_overfitting(self, actions, instruction, emit) -> None:
        if not instruction:
            return
        instruction_lc = instruction.lower()
        for entry in actions:
            action = entry.payload.get("action") or {}
            for value in _string_values(action):
                candidate = value.strip().lower()
                if len(candidate) >= self._min_overfit_chars and candidate in instruction_lc:
                    if _result_is_empty(entry.payload.get("result")):
                        emit(
                            FailureMode.SURFACE_OVERFITTING,
                            f"searched for a verbatim slice of the instruction "
                            f"({value!r}) which returned nothing",
                        )
                        return

    # ------------------------------------------------------------------

    def _final_answer(self, llm_calls) -> Any:
        for entry in reversed(llm_calls):
            response = entry.payload.get("response")
            if response is not None:
                return response
        return None


def cluster_failure_modes(reports: list[RunLossReport], top_n: int = 5) -> list[FailureCluster]:
    """Aggregate per-run reports into per-mode clusters.

    Reuses :class:`~forge.runtime.clustering.FailureCluster` and mirrors
    ``FailureClusterer``'s convention (buckets sorted most-frequent first, up to
    five sample ids each). The cluster's ``check_name`` holds the failure-mode
    value so mode aggregation and check aggregation share one shape.
    """
    buckets: dict[str, list[str]] = {}
    for report in reports:
        for mode in dict.fromkeys(report.mode_names):  # de-dupe per run
            buckets.setdefault(mode, []).append(report.run_id)

    clusters = [
        FailureCluster(check_name=mode, count=len(ids), episode_ids=ids[:5])
        for mode, ids in buckets.items()
    ]
    clusters.sort(key=lambda c: c.count, reverse=True)
    return clusters[:top_n]
