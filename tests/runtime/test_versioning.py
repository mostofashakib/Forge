"""Tests for Artifact Versioning, Test Isolation, and Baseline Test Suites."""
from __future__ import annotations

from forge.contracts.types import CheckResult, Task, VerificationResult
from forge.contracts.versioning import (
    ArtifactType,
    InitialTest,
    InitialTestSuite,
    VersionedArtifact,
    VersionProvenance,
)
from forge.runtime.episode_grader import EpisodeGrader
from forge.runtime.reward import RewardEngine
from forge.runtime.snapshot import StepSnapshot
from forge.runtime.trajectory import Trajectory
from forge.runtime.verifier import FunctionVerifier, VerifierEngine
from forge.runtime.versioning.grading import IsolatedGrader
from forge.runtime.versioning.isolation import TestIsolationBoundary
from forge.runtime.versioning.registry import ArtifactVersionRegistry


def make_step(index: int, command: str | None = None) -> StepSnapshot:
    return StepSnapshot(
        episode_id="ep_ver_test",
        step_index=index,
        state_hash_before="h0",
        state_hash_after="h1",
        action={"type": "bash", "params": {"command": command or "echo hello"}},
        events=[],
        reward=0.0,
        verifier_results=[],
        diff={"added": {}, "changed": {}, "removed": {}},
        terminated=False,
        truncated=False,
    )


# ==============================================================================
# Artifact Version Registry Tests
# ==============================================================================

def test_registry_registers_and_hashes_artifacts():
    registry = ArtifactVersionRegistry()

    prompt_art = registry.register_prompt(
        name="cot_system_prompt",
        template="Think step by step and answer concisely.",
        version_tag="v1.2.0",
        metadata={"model": "claude-3-5-sonnet"},
    )
    assert prompt_art.artifact_type == ArtifactType.PROMPT
    assert prompt_art.version_tag == "v1.2.0"
    assert len(prompt_art.version_hash) == 64

    # Identical content yields identical hash
    hash2 = registry.compute_hash("Think step by step and answer concisely.")
    assert hash2 == prompt_art.version_hash

    # Modified content yields different hash
    hash3 = registry.compute_hash("Think step by step and be verbose.")
    assert hash3 != prompt_art.version_hash


def test_registry_diff_versions_for_debugging():
    registry = ArtifactVersionRegistry()
    v1 = registry.register_prompt("sys_prompt", "Prompt version 1", version_tag="v1.0.0", metadata={"temp": 0.0})
    v2 = registry.register_prompt("sys_prompt", "Prompt version 2", version_tag="v2.0.0", metadata={"temp": 0.7})

    diff = registry.diff_versions(v1, v2)
    assert diff["name"] == "sys_prompt"
    assert diff["identical"] is False
    assert diff["metadata_diff"]["changed"]["temp"] == {"from": 0.0, "to": 0.7}


def test_registry_creates_cryptographic_provenance():
    registry = ArtifactVersionRegistry()
    provenance = registry.create_provenance(
        prompt="system: execute safely",
        verifier={"rules": {"exit_code": 0}},
        data={"users": [{"id": 1}]},
        test_suite={"suite_id": "init_suite_1"},
    )
    assert isinstance(provenance, VersionProvenance)
    assert len(provenance.prompt_version) == 16
    assert len(provenance.verifier_version) == 16
    assert len(provenance.data_version) == 16
    assert provenance.test_suite_version is not None
    assert len(provenance.composite_digest) == 16


# ==============================================================================
# Test Isolation Boundary Tests
# ==============================================================================

def test_isolation_scrubs_test_keys_from_observation():
    untrusted_obs = {
        "user_balance": 100,
        "current_page": "dashboard",
        "tests": ["test_payment_gateway", "test_fraud_filter"],
        "unit_tests": {"assert_balance": 100},
        "pytest_passed": True,
        "ground_truth": {"expected_secret": "xyz"},
        "nested": {
            "visible_key": "ok",
            "hidden_tests": "forbidden",
        },
    }

    sanitized = TestIsolationBoundary.sanitize_observation(untrusted_obs)

    assert "tests" not in sanitized
    assert "unit_tests" not in sanitized
    assert "pytest_passed" not in sanitized
    assert "ground_truth" not in sanitized
    assert "hidden_tests" not in sanitized["nested"]
    assert sanitized["user_balance"] == 100
    assert sanitized["nested"]["visible_key"] == "ok"


def test_isolation_scrubs_internal_criteria_from_task():
    task = Task(
        id="order_submission",
        objective="Submit an order for 2 widgets",
        success_conditions=[{"key": "order_placed", "value": True}],
        failure_conditions=[{"key": "canceled", "value": True}],
        metadata={"unit_tests": ["test_order_widget"], "target_sku": "WIDGET-01"},
    )

    agent_task = TestIsolationBoundary.sanitize_task_for_agent(task)
    assert agent_task["id"] == "order_submission"
    assert agent_task["objective"] == "Submit an order for 2 widgets"
    assert "unit_tests" not in agent_task["metadata"]
    assert agent_task["metadata"]["target_sku"] == "WIDGET-01"
    assert "success_conditions" not in agent_task


def test_isolation_detects_unauthorized_test_inspection():
    traj = Trajectory(
        episode_id="ep_breach",
        steps=[
            make_step(0, command="ls -la"),
            make_step(1, command="cat tests/test_payment.py"),
        ],
    )
    breaches = TestIsolationBoundary.audit_agent_isolation(traj)
    assert len(breaches) == 1
    assert "Agent attempted unauthorized inspection of test suite" in breaches[0]


# ==============================================================================
# Initial Test Suite & Isolated Grader Tests
# ==============================================================================

def test_initial_test_suite_execution():
    suite = InitialTestSuite(
        suite_id="baseline_shop",
        tests=[
            InitialTest(
                id="check_order_status",
                name="Order status confirmed",
                assertions=[{"key": "order_status", "expected": "confirmed"}],
            ),
            InitialTest(
                id="check_inventory_decremented",
                name="Inventory decremented",
                assertions=[{"key": "inventory_count", "expected": 9}],
            ),
        ],
    )
    # State satisfying both tests
    state = {"order_status": "confirmed", "inventory_count": 9}
    checks = suite.execute_isolated(state, Trajectory(episode_id="ep", steps=[]))
    assert len(checks) == 2
    assert all(c.passed for c in checks)


def test_isolated_grader_evaluates_initial_tests_and_enforces_isolation():
    initial_suite = InitialTestSuite(
        suite_id="init_1",
        tests=[
            InitialTest(
                id="test_payment",
                name="Payment processed",
                assertions=[{"key": "paid", "expected": True}],
            )
        ],
    )
    v_engine = VerifierEngine()
    v_engine.register(
        "mock_v",
        FunctionVerifier(lambda s, tr, t: VerificationResult.from_checks("mock_v", [CheckResult(name="v_check", passed=True, score=1.0)])),
    )
    grader = IsolatedGrader(
        verifier_engine=v_engine,
        initial_test_suite=initial_suite,
    )

    clean_traj = Trajectory(episode_id="ep", steps=[make_step(0, "submit_payment")])
    eval_result = grader.grade(
        state={"paid": True},
        trajectory=clean_traj,
        task={"verifier_id": "mock_v"},
    )
    assert eval_result.passed is True
    assert any("initial_tests" in r.verifier_id for r in eval_result.verification_results)


def test_isolated_grader_fails_on_test_access_breach():
    grader = IsolatedGrader()
    # Trajectory where agent cheated by catting the test files
    cheating_traj = Trajectory(
        episode_id="ep",
        steps=[make_step(0, "cat tests/test_grader.py")],
    )
    eval_result = grader.grade(
        state={"paid": True},
        trajectory=cheating_traj,
        task={},
    )
    assert eval_result.passed is False
    assert eval_result.total_reward == 0.0
    assert "isolation_breach" in eval_result.reason


def test_episode_grader_delegates_to_isolated_grader():
    v_engine = VerifierEngine()
    v_engine.register(
        "simple_v",
        FunctionVerifier(lambda s, tr, t: VerificationResult.from_checks("simple_v", [CheckResult(name="pass", passed=True, score=1.0)])),
    )
    r_engine = RewardEngine()
    ep_grader = EpisodeGrader(verifier_engine=v_engine, reward_engine=r_engine)

    eval_result = ep_grader.evaluate(
        state={},
        trajectory=Trajectory(episode_id="ep", steps=[make_step(0, "echo clean")]),
        task={"verifier_id": "simple_v"},
        invalid_action_count=0,
        reason="completed",
    )
    assert eval_result.passed is True
