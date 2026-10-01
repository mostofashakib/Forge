"""Grade an in-process episode exactly once: verifiers decide, the rubric scores."""
from __future__ import annotations

from forge.contracts import EpisodeEvaluation
from forge.runtime.reward import RewardEngine
from forge.runtime.verifier import VerifierEngine


class EpisodeGrader:
    """Produces the episode's single authoritative verdict and remembers it.

    Repeated calls return the first verdict, so a step that terminates and a
    later `finalize_episode` can never grade the same episode twice.
    """

    def __init__(self, verifier_engine: VerifierEngine, reward_engine: RewardEngine) -> None:
        self._verifiers = verifier_engine
        self._rubric = reward_engine
        self._evaluation: EpisodeEvaluation | None = None

    @property
    def evaluation(self) -> EpisodeEvaluation | None:
        """The verdict, or None while the episode is still open."""
        return self._evaluation

    def reset(self) -> None:
        self._evaluation = None

    def evaluate(
        self, state: dict, trajectory, task: dict | None, *, invalid_action_count: int, reason: str,
    ) -> EpisodeEvaluation:
        if self._evaluation is not None:
            return self._evaluation
        verifier_results = self._verifiers.run_all(state, trajectory, task)
        reward = self._rubric.compute(
            state,
            trajectory,
            verifier_results,
            {**(task or {}), "invalid_action_count": invalid_action_count},
        )
        self._evaluation = EpisodeEvaluation(
            passed=any(result.passed for result in verifier_results),
            reward=reward,
            verification_results=verifier_results,
            reason=reason,
        )
        return self._evaluation
