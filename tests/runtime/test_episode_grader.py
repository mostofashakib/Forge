"""EpisodeGrader verifies and scores an episode once, and only once."""
from __future__ import annotations

from types import SimpleNamespace

from forge.runtime.episode_grader import EpisodeGrader
from forge.runtime.reward import RewardEngine
from forge.runtime.verification import CheckResult, VerificationResult
from forge.runtime.verifier import FunctionVerifier, VerifierEngine


def _grader(passes: bool, calls: list) -> EpisodeGrader:
    def verifier(state, trajectory, task):
        calls.append(task)
        return VerificationResult.from_checks(
            "v", [CheckResult(name="c", passed=passes, score=1.0 if passes else 0.0)]
        )

    engine = VerifierEngine()
    engine.register("v", FunctionVerifier(verifier))
    return EpisodeGrader(engine, RewardEngine())


def test_the_first_verdict_is_final_until_reset():
    calls: list = []
    grader = _grader(True, calls)
    trajectory = SimpleNamespace(steps=[])

    task = {"id": "t", "verifier_id": "v"}

    first = grader.evaluate({}, trajectory, task, invalid_action_count=2, reason="submitted")
    second = grader.evaluate({}, trajectory, task, invalid_action_count=0, reason="max_steps")

    assert second is first
    assert (first.passed, first.reason) == (True, "submitted")
    assert calls == [task]


def test_a_failing_verifier_fails_the_episode_and_reset_reopens_it():
    calls: list = []
    grader = _grader(False, calls)

    task = {"id": "t", "verifier_id": "v"}
    assert grader.evaluate({}, SimpleNamespace(steps=[]), task, invalid_action_count=0, reason="r").passed is False
    grader.reset()
    assert grader.evaluation is None
