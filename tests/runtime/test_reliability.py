import pytest
from forge.runtime.reliability import (
    InfrastructureCrash,
    AgentFailure,
    DivergenceError,
    classify_failure,
    compute_environment_version,
    execute_reliable_episode,
    SnapshotManager,
    AttemptRecord,
    INFRASTRUCTURE_REASONS,
    AGENT_REASONS,
)
from forge.contracts.termination import BudgetTerminationPolicy, BUDGET_REASONS
from forge.contracts.types import StepOutcome


def test_classify_failure_infrastructure():
    crash = InfrastructureCrash("container_unreachable", "Connection refused")
    ftype, freason = classify_failure(crash)
    assert ftype == "infrastructure"
    assert freason == "container_unreachable"


def test_classify_failure_agent():
    fail = AgentFailure("invalid_action", "Action not in schema")
    ftype, freason = classify_failure(fail)
    assert ftype == "agent"
    assert freason == "invalid_action"


def test_classify_failure_generic():
    exc = ConnectionError("Connection dropped by peer")
    ftype, freason = classify_failure(exc)
    assert ftype == "infrastructure"
    assert freason == "transport_timeout"


def test_snapshot_and_replay_divergence():
    mgr = SnapshotManager(snapshot_interval=2)
    # Step 1: record snapshot and model output
    mgr.record_snapshot(1, {"state_key": "val1"})
    mgr.record_model_output(1, {"tool": "act1", "args": {}})

    # Step 2: record snapshot and model output
    mgr.record_snapshot(2, {"state_key": "val2"})
    mgr.record_model_output(2, {"tool": "act2", "args": {}})

    assert mgr.latest_snapshot_step() == 2
    assert mgr.get_snapshot(2) == {"state_key": "val2"}

    # Divergence verification: matching state hash passes
    mgr.verify_divergence(step=1, expected_hash="hash1", actual_hash="hash1")

    # Mismatch raises DivergenceError
    with pytest.raises(DivergenceError) as exc_info:
        mgr.verify_divergence(step=1, expected_hash="hash1", actual_hash="hash_mismatch")
    assert "diverged at step 1" in str(exc_info.value)


def test_execute_reliable_episode_retry_and_recovery():
    attempts_called = 0

    def mock_runner(attempt: int, seed: int):
        nonlocal attempts_called
        attempts_called += 1
        if attempts_called < 2:
            raise InfrastructureCrash("container_unreachable", "Simulated container blip")
        return {"status": "ok", "total_reward": 1.0}

    result, attempts = execute_reliable_episode(
        task_runner=mock_runner,
        task_id="task_test_01",
        env_name="test_env",
        environment_version="pkg_test_123",
        seed=42,
    )

    assert result["status"] == "ok"
    assert len(attempts) == 2
    assert attempts[0].error is not None
    assert attempts[1].error is None


def test_execute_reliable_episode_agent_failure_does_not_retry():
    attempts_called = 0

    def mock_agent_fail(attempt: int, seed: int):
        nonlocal attempts_called
        attempts_called += 1
        raise AgentFailure("policy_violation", "Agent violated safety policy")

    with pytest.raises(AgentFailure):
        execute_reliable_episode(
            task_runner=mock_agent_fail,
            task_id="task_agent_01",
            env_name="test_env",
            environment_version="pkg_test_123",
            seed=42,
        )

    # Agent failures fail immediately without consuming infrastructure retries
    assert attempts_called == 1


def test_execute_reliable_episode_quarantine_on_all_crashes():
    attempts_called = 0

    def mock_always_crash(attempt: int, seed: int):
        nonlocal attempts_called
        attempts_called += 1
        raise InfrastructureCrash("reset_failed", "Container crashed on reset")

    with pytest.raises(InfrastructureCrash):
        execute_reliable_episode(
            task_runner=mock_always_crash,
            task_id="task_crash_01",
            env_name="test_env",
            environment_version="pkg_test_123",
            seed=42,
        )

    assert attempts_called == 4  # 1 initial attempt + 3 retries (retry_cap = 3)


def test_budget_truncation_policy():
    policy = BudgetTerminationPolicy(
        max_steps=5,
        max_tokens=100,
        max_wall_clock_time=10.0,
        max_cost=1.0,
    )

    # Within budget: step outcome
    step1 = StepOutcome(step_index=1, reward=0.1, tokens=20, wall_clock_time=1.0, cost=0.1)
    res1 = policy.check(step1)
    assert res1 is None

    # Exceed steps budget
    step_exceed = StepOutcome(step_index=5, reward=0.1, tokens=30, wall_clock_time=2.0, cost=0.2)
    res = policy.check(step_exceed)

    assert res is not None
    assert res.truncated
    assert res.reason == "max_steps"
    assert res.reason in BUDGET_REASONS


def test_environment_version_ignores_run_outputs(tmp_path):
    from forge.runtime.reliability import compute_environment_version

    env_dir = tmp_path / "ledger"
    (env_dir / "transitions").mkdir(parents=True)
    (env_dir / "gym_wrapper.py").write_text("def build(): ...\n")
    before = compute_environment_version("ledger", env_type="in_process", env_dir=env_dir)

    # Everything a run writes beside the package.
    for output in ("episodes/ep_1.jsonl", "episodes/ep_1.trace.jsonl", "agent_episodes/cep_1.jsonl",
                   "exports/sft.jsonl", "__pycache__/gym_wrapper.cpython-311.pyc", "port"):
        (env_dir / output).parent.mkdir(parents=True, exist_ok=True)
        (env_dir / output).write_text("x")

    assert compute_environment_version("ledger", env_type="in_process", env_dir=env_dir) == before


def test_environment_version_changes_when_the_package_changes(tmp_path):
    from forge.runtime.reliability import compute_environment_version

    env_dir = tmp_path / "ledger"
    env_dir.mkdir()
    (env_dir / "gym_wrapper.py").write_text("def build(): ...\n")
    before = compute_environment_version("ledger", env_type="in_process", env_dir=env_dir)

    (env_dir / "gym_wrapper.py").write_text("def build(): return 1\n")

    assert compute_environment_version("ledger", env_type="in_process", env_dir=env_dir) != before


def test_a_type_error_inside_the_runner_is_not_mistaken_for_a_signature_mismatch(monkeypatch):
    # The runner's own bug must surface, not trigger re-invoking the episode
    # with other argument shapes inside the same attempt.
    monkeypatch.setenv("FORGE_RETRY_CAP", "0")
    calls = []

    def buggy_runner(attempt: int, seed: int):
        calls.append((attempt, seed))
        raise TypeError("unsupported operand type(s)")

    with pytest.raises(TypeError, match="unsupported operand"):
        execute_reliable_episode(
            task_runner=buggy_runner, task_id="t", env_name="e", environment_version="v", seed=7,
        )

    assert calls == [(0, 7)]
