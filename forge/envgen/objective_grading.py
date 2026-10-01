"""Grade a container episode once: the LLM objective judge decides, the rubric scores."""
from __future__ import annotations

from typing import TYPE_CHECKING

from forge.contracts import EpisodeEvaluation, Rubric, Task
from forge.envgen.objective import ObjectiveScorer, objective_verification
from forge.runtime.trajectory import Trajectory

if TYPE_CHECKING:
    from forge.envgen.episode_runner import EpisodeResult


class ObjectiveGrader:
    """Issues the single authoritative verdict for an episode and applies it to the result."""

    def __init__(self, scorer: ObjectiveScorer, *, objective: str, success_threshold: float) -> None:
        self._scorer = scorer
        self._objective = objective
        self._threshold = success_threshold

    def grade(
        self,
        result: EpisodeResult,
        state: dict,
        action: dict,
        *,
        task: Task,
        rubric: Rubric,
        derived_diff: dict | None = None,
        state_changed: bool = False,
    ) -> None:
        """Score `state`, count the verdict, and record pass/fail and reward on `result`.

        A judge that raises leaves `result` untouched, so a failed grade is never
        mistaken for a zero score.
        """
        score = self._scorer.score(
            state, self._objective, derived_diff=derived_diff, action_taken=action,
        )
        result.llm_verdicts += 1
        verification = objective_verification(score, self._threshold)
        graded_task = task.model_copy(update={
            "metadata": {**task.metadata, "state_changed": state_changed},
        })
        reward = rubric.score(
            state,
            Trajectory(episode_id=result.episode_id, steps=[]),
            [verification],
            graded_task,
        )
        result.final_objective_score = score
        result.apply_evaluation(EpisodeEvaluation(
            passed=verification.passed,
            reward=reward,
            verification_results=[verification],
            reason=result.termination_reason,
        ))
