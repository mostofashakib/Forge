from forge.runtime.verification import CheckResult, VerificationResult
from forge.runtime.verifier import VerifierEngine


def passing_verifier(state, trajectory, task):
    return VerificationResult.from_checks(
        "always_pass", [CheckResult(name="check", passed=True, score=1.0)]
    )


def failing_verifier(state, trajectory, task):
    return VerificationResult.from_checks(
        "always_fail",
        [CheckResult(name="check", passed=False, score=0.0, evidence="never passes")],
    )


def test_registered_verifier_runs():
    engine = VerifierEngine()
    engine.register("always_pass", passing_verifier)
    results = engine.run_all({}, None, {"name": "t", "verifier_id": "always_pass"})
    assert len(results) == 1
    assert results[0].passed is True


def test_no_task_returns_empty_results():
    engine = VerifierEngine()
    engine.register("v", passing_verifier)
    assert engine.run_all({}, None, None) == []


def test_unknown_verifier_id_returns_empty():
    engine = VerifierEngine()
    results = engine.run_all({}, None, {"name": "t", "verifier_id": "nonexistent"})
    assert results == []


def test_failing_verifier_result_has_passed_false():
    engine = VerifierEngine()
    engine.register("always_fail", failing_verifier)
    results = engine.run_all({}, None, {"name": "t", "verifier_id": "always_fail"})
    assert results[0].passed is False
    assert results[0].checks[0].evidence == "never passes"
