"""Compose a multi-tiered :class:`LayeredVerifier` per environment/task and
score its results (TASKS.md #4).

The runtime already provides every tier (state, invariant, trajectory, judge,
negative) plus :class:`RewardBreakdown` for partial scoring. This module is the
missing composer: it maps a task's declared success/failure conditions and the
scenario ground truth onto those tiers, and offers a clean binary-vs-partial
scoring choice. It reuses the existing primitives — it does not reimplement them.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from forge.extraction.schemas import CompilerInput, SuccessCondition, TaskTemplate
from forge.runtime.layered_verifier import JudgeClient, LayeredVerifier
from forge.runtime.reward import RewardBreakdown, RewardComponent
from forge.runtime.verifiers.exact_state import ExactStateVerifier
from forge.runtime.verification import VerificationResult


@runtime_checkable
class ScenarioLike(Protocol):
    """Structural view of a scenario's trajectory ground truth."""

    required_actions: list[str]
    forbidden_actions: list[str]
    ordering_sensitive: bool


class ScoringMode(str, Enum):
    BINARY = "binary"
    PARTIAL = "partial"


# A comma-separated expression lists several event/action names in one condition.
def _names(expression: str) -> list[str]:
    return [part.strip() for part in expression.split(",") if part.strip()]


class VerifierComposer:
    """Builds a configured :class:`LayeredVerifier` and scores its results."""

    def compose(
        self,
        task: TaskTemplate,
        scenario: ScenarioLike | None = None,
        judge_client: JudgeClient | None = None,
        verifier_id: str | None = None,
    ) -> LayeredVerifier:
        verifier = LayeredVerifier(verifier_id or task.name, judge_client=judge_client)
        for index, condition in enumerate(task.success_conditions):
            self._apply_success(verifier, condition, index)
        for index, condition in enumerate(task.failure_conditions):
            self._apply_failure(verifier, condition, index)
        if scenario is not None:
            self._apply_scenario(verifier, scenario)
        return verifier

    def compose_all(
        self,
        compiler_input: CompilerInput,
        judge_client: JudgeClient | None = None,
    ) -> dict[str, LayeredVerifier]:
        """Compose one verifier per task in an environment, keyed by task name."""
        return {
            task.name: self.compose(task, judge_client=judge_client)
            for task in compiler_input.tasks
        }

    # ------------------------------------------------------------------
    # Condition → tier mapping
    # ------------------------------------------------------------------

    def _apply_success(self, verifier: LayeredVerifier, condition: SuccessCondition, i: int) -> None:
        kind = condition.type
        if kind == "state_check":
            checker = ExactStateVerifier(condition.expression)
            verifier.add_state_check(f"state_{i}", checker.check)
        elif kind == "event_check":
            verifier.require_milestones(_names(condition.expression), ordered=False)
        elif kind == "temporal_check":
            verifier.require_milestones(_names(condition.expression), ordered=True)
        elif kind == "policy_check":
            # A satisfied policy means the forbidden actions were never taken.
            verifier.forbid_actions(_names(condition.expression))
        elif kind == "semantic_check":
            rubric = condition.rubric or condition.expression
            verifier.add_llm_judge(f"semantic_{i}", rubric)
        elif kind == "negative_check":
            verifier.forbid_events(_names(condition.expression))
        else:
            raise ValueError(f"Unknown success condition type: {kind!r}")

    def _apply_failure(self, verifier: LayeredVerifier, condition: SuccessCondition, i: int) -> None:
        """A failure condition describes something that must NOT hold."""
        kind = condition.type
        if kind in ("event_check", "negative_check", "temporal_check"):
            verifier.forbid_events(_names(condition.expression))
        elif kind == "policy_check":
            verifier.forbid_actions(_names(condition.expression))
        elif kind == "state_check":
            expression = condition.expression
            checker = ExactStateVerifier(expression)

            def check(state, trajectory, task, _checker=checker, _expr=expression):
                inner = _checker.check(state, trajectory, task)
                # The failure state must NOT be reached: invert the outcome.
                passed = not inner.passed
                return passed, None if passed else f"failure state reached: {_expr}"

            verifier.add_negative_check(f"forbidden_state_{i}", check)
        else:
            raise ValueError(f"Unknown failure condition type: {kind!r}")

    def _apply_scenario(self, verifier: LayeredVerifier, scenario: ScenarioLike) -> None:
        required = list(scenario.required_actions)
        if required:
            if scenario.ordering_sensitive:
                verifier.require_action_sequence(required)
            else:
                verifier.require_actions(required)
        if scenario.forbidden_actions:
            verifier.forbid_actions(list(scenario.forbidden_actions))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        result: VerificationResult,
        mode: ScoringMode = ScoringMode.BINARY,
        weights: dict[str, float] | None = None,
    ) -> RewardBreakdown:
        """Turn a verification result into reward points.

        BINARY: full credit only when every tier passed, otherwise zero.
        PARTIAL: weighted mean of per-tier scores, so a partially-correct
        trajectory still earns graded credit.
        """
        if mode == ScoringMode.BINARY:
            value = 1.0 if result.passed else 0.0
            return RewardBreakdown(
                total_reward=value,
                components=[RewardComponent(name="task_success", value=value)],
            )

        tier_scores: dict[str, list[float]] = {}
        for check in result.checks:
            tier = check.name.split(":", 1)[0]
            tier_scores.setdefault(tier, []).append(check.score)

        components: list[RewardComponent] = []
        weighted_sum = 0.0
        weight_total = 0.0
        for tier, scores in tier_scores.items():
            tier_score = sum(scores) / len(scores)
            weight = (weights or {}).get(tier, 1.0)
            components.append(RewardComponent(name=tier, value=tier_score))
            weighted_sum += weight * tier_score
            weight_total += weight

        total = weighted_sum / weight_total if weight_total else 0.0
        return RewardBreakdown(total_reward=total, components=components)
