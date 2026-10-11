from forge.contracts import EpisodeEvaluation
from forge.runtime.reward import RewardEngine
from forge.runtime.verifier import VerifierEngine
from forge.runtime.versioning.grading import IsolatedGrader


class EpisodeGrader:
    """Produces the episode's single authoritative verdict and remembers it.

    Evaluates against the authoritative initial test suite, enforces complete
    agent test isolation, and records cryptographic version provenance.
    """

    def __init__(
        self,
        verifier_engine: VerifierEngine,
        reward_engine: RewardEngine,
        isolated_grader: IsolatedGrader | None = None,
    ) -> None:
        self._verifiers = verifier_engine
        self._rubric = reward_engine
        self._isolated_grader = isolated_grader or IsolatedGrader(
            verifier_engine=verifier_engine, reward_engine=reward_engine
        )
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
        self._evaluation = self._isolated_grader.grade(
            state,
            trajectory,
            task,
            invalid_action_count=invalid_action_count,
            reason=reason,
        )
        return self._evaluation
