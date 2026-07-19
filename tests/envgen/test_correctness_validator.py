import httpx
import pytest
import respx

from forge.envgen.correctness_validator import (
    CorrectnessValidationError,
    CorrectnessValidationResult,
    CorrectnessFinding,
    CorrectnessValidator,
)


def _state_sequence(states):
    it = iter(states)

    def _side_effect(request):
        return httpx.Response(200, json=next(it))

    return _side_effect


def _mock_all_routes():
    respx.post("http://c/forge/reset").respond(200, json={"ok": True})
    respx.post("http://c/add_todo").respond(200, json={"ok": True})
    respx.post("http://c/forge/snapshot").respond(200, json={"ok": True})
    respx.post("http://c/forge/restore/correctness").respond(200, json={"ok": True})


@respx.mock
def test_reset_fidelity_passes_when_state_returns_to_baseline():
    baseline = {"todos": [{"id": 1, "created_at": 0}]}
    _mock_all_routes()
    # 5 checkpoints, all baseline: idempotent reset, reset restores baseline,
    # snapshot captures baseline, restore reproduces baseline.
    respx.get("http://c/forge/state").mock(side_effect=_state_sequence([
        baseline, baseline, baseline, baseline, baseline,
    ]))
    result = CorrectnessValidator(base_url="http://c").validate(["add_todo"])
    assert result.passed is True
    assert result.findings == []


@respx.mock
def test_reset_fidelity_fails_when_state_drifts():
    baseline = {"todos": [{"id": 1, "created_at": 0}]}
    drifted = {"todos": [{"id": 1, "created_at": 5}]}  # wall-clock drift on reseed
    _mock_all_routes()
    # Checkpoint 3 (after mutate->reset) drifts; snapshot/restore still clean.
    respx.get("http://c/forge/state").mock(side_effect=_state_sequence([
        baseline, baseline, drifted, baseline, baseline,
    ]))
    result = CorrectnessValidator(base_url="http://c").validate(["add_todo"])
    assert result.passed is False
    assert any(f.category == "reset_fidelity" for f in result.findings)


@respx.mock
def test_snapshot_restore_round_trip_fails_when_lossy():
    baseline = {"todos": [{"id": 1, "created_at": 0}]}
    mutated = {"todos": [{"id": 1, "created_at": 0}, {"id": 2, "created_at": 1}]}
    _mock_all_routes()
    # Reset fidelity passes (checkpoints 1-3 baseline); checkpoint 5 (after
    # restore) does not match the snapshot captured at checkpoint 4.
    respx.get("http://c/forge/state").mock(side_effect=_state_sequence([
        baseline, baseline, baseline, baseline, mutated,
    ]))
    result = CorrectnessValidator(base_url="http://c").validate(["add_todo"])
    assert result.passed is False
    assert any(f.category == "snapshot_restore" for f in result.findings)


def test_validation_error_carries_result():
    result = CorrectnessValidationResult(
        passed=False, findings=[CorrectnessFinding("reset_fidelity", "drift")],
    )
    err = CorrectnessValidationError(result)
    assert err.result is result
    assert "drift" in str(err)
