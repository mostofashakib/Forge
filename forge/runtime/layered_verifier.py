from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from forge.runtime.errors import VerifierConfigurationError
from forge.runtime.verification import CheckResult, VerificationResult

# A check callable takes (state, trajectory, task) and returns one of:
#   bool, (bool, evidence-str), or a CheckResult.
CheckFn = Callable[[dict, object, dict], object]

# A judge client takes (rubric, state, trajectory, task) and returns
# (score in [0, 1], reasoning).
JudgeClient = Callable[[str, dict, object, dict], tuple[float, str]]


@dataclass
class _Check:
    layer: str
    name: str
    fn: CheckFn


@dataclass
class _JudgeCheck:
    name: str
    rubric: str
    threshold: float


class LayeredVerifier:
    """Five-layer verifier for agent episodes, runnable under VerifierEngine.

    Layers run in order:
      state      — does the final state hold the required changes?
      invariant  — were all intermediate milestones reached, none skipped?
      trajectory — were the necessary tool calls made, and no unnecessary ones?
      judge      — LLM-as-a-judge against a rubric, for creative tasks
      negative   — no unintended side effects (deletions, redundant calls, …)
    """

    LAYERS = ("state", "invariant", "trajectory", "judge", "negative")

    def __init__(self, verifier_id: str, judge_client: JudgeClient | None = None) -> None:
        self.verifier_id = verifier_id
        self._judge_client = judge_client
        self._checks: dict[str, list[_Check]] = {layer: [] for layer in self.LAYERS}
        self._judges: list[_JudgeCheck] = []
        self.required_milestones: list[str] = []

    # ------------------------------------------------------------------
    # Layer registration
    # ------------------------------------------------------------------

    def add_state_check(self, name: str, fn: CheckFn) -> "LayeredVerifier":
        self._checks["state"].append(_Check("state", name, fn))
        return self

    def add_invariant_check(self, name: str, fn: CheckFn) -> "LayeredVerifier":
        self._checks["invariant"].append(_Check("invariant", name, fn))
        return self

    def add_trajectory_check(self, name: str, fn: CheckFn) -> "LayeredVerifier":
        self._checks["trajectory"].append(_Check("trajectory", name, fn))
        return self

    def add_llm_judge(self, name: str, rubric: str, threshold: float = 0.7) -> "LayeredVerifier":
        self._judges.append(_JudgeCheck(name, rubric, threshold))
        return self

    def add_negative_check(self, name: str, fn: CheckFn) -> "LayeredVerifier":
        self._checks["negative"].append(_Check("negative", name, fn))
        return self

    # ------------------------------------------------------------------
    # Convenience helpers per layer
    # ------------------------------------------------------------------

    def require_milestones(self, event_types: list[str], ordered: bool = True) -> "LayeredVerifier":
        """Invariant: every milestone event occurred, in order unless ordered=False."""
        self.required_milestones.extend(event_types)

        def check(state, trajectory, task):
            events = [e.get("type") for e in trajectory.events]
            positions = []
            for milestone in event_types:
                if milestone not in events:
                    return False, f"milestone '{milestone}' never reached"
                positions.append(events.index(milestone))
            if ordered and positions != sorted(positions):
                return False, f"milestones reached out of order: {event_types}"
            return True, None

        return self.add_invariant_check("milestones", check)

    def require_actions(self, action_types: list[str]) -> "LayeredVerifier":
        """Trajectory: every listed tool call was made at least once."""

        def check(state, trajectory, task):
            taken = {step.action.get("type") for step in trajectory.steps}
            missing = [a for a in action_types if a not in taken]
            if missing:
                return False, f"required actions never taken: {missing}"
            return True, None

        return self.add_trajectory_check("required_actions", check)

    def forbid_actions(self, action_types: list[str]) -> "LayeredVerifier":
        """Trajectory: none of the listed tool calls were made."""

        def check(state, trajectory, task):
            taken = [step.action.get("type") for step in trajectory.steps]
            offending = [a for a in taken if a in set(action_types)]
            if offending:
                return False, f"unnecessary actions taken: {offending}"
            return True, None

        return self.add_trajectory_check("forbidden_actions", check)

    def forbid_events(self, event_types: list[str]) -> "LayeredVerifier":
        """Negative: none of the listed side-effect events were emitted."""

        def check(state, trajectory, task):
            emitted = [e.get("type") for e in trajectory.events if e.get("type") in set(event_types)]
            if emitted:
                return False, f"unintended side effects: {emitted}"
            return True, None

        return self.add_negative_check("forbidden_events", check)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def __call__(self, state: dict, trajectory, task: dict) -> VerificationResult:
        checks: list[CheckResult] = []
        for layer in self.LAYERS:
            if layer == "judge":
                checks.extend(self._run_judges(state, trajectory, task))
                continue
            for check in self._checks[layer]:
                checks.append(self._run_check(check, state, trajectory, task))
        result = VerificationResult.from_checks(self.verifier_id, checks)
        failed_layers = sorted(
            {c.name.split(":", 1)[0] for c in checks if not c.passed},
            key=self.LAYERS.index,
        )
        if failed_layers:
            result.explanation = f"failed layers: {', '.join(failed_layers)}"
        return result

    def _run_check(self, check: _Check, state, trajectory, task) -> CheckResult:
        outcome = check.fn(state, trajectory, task)
        name = f"{check.layer}:{check.name}"
        if isinstance(outcome, CheckResult):
            return outcome.model_copy(update={"name": name})
        evidence = None
        if isinstance(outcome, tuple):
            outcome, evidence = outcome
        passed = bool(outcome)
        return CheckResult(
            name=name,
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=evidence,
        )

    def _run_judges(self, state, trajectory, task) -> list[CheckResult]:
        if not self._judges:
            return []
        if self._judge_client is None:
            raise VerifierConfigurationError(
                f"LayeredVerifier '{self.verifier_id}' has LLM judge checks but no judge_client"
            )
        results = []
        for judge in self._judges:
            score, reasoning = self._judge_client(judge.rubric, state, trajectory, task)
            results.append(
                CheckResult(
                    name=f"judge:{judge.name}",
                    passed=score >= judge.threshold,
                    score=score,
                    evidence=reasoning,
                )
            )
        return results
