"""Tests for Verifier Trust Boundary and Domain Group Adapters (CUA, TU, Code, Data)."""
from __future__ import annotations

from forge.contracts.types import CheckResult, Task, VerificationResult
from forge.contracts.verification import (
    GroupVerificationSpec,
    TrustBoundaryReport,
    TrustLevel,
    VerifierGroup,
)
from forge.runtime.snapshot import StepSnapshot
from forge.runtime.trajectory import Trajectory
from forge.runtime.trust_boundary import (
    CUAAdapter,
    CodeAdapter,
    DataAdapter,
    TUAdapter,
    VerifierTrustBoundary,
)
from forge.runtime.verifier import FunctionVerifier, VerifierEngine


def make_step(
    index: int,
    action_type: str = "noop",
    params: dict | None = None,
    events: list[dict] | None = None,
    before_hash: str = "h1",
    after_hash: str = "h2",
) -> StepSnapshot:
    return StepSnapshot(
        episode_id="ep_trust_test",
        step_index=index,
        state_hash_before=before_hash,
        state_hash_after=after_hash,
        action={"type": action_type, "params": params or {}},
        events=events or [],
        reward=0.0,
        verifier_results=[],
        diff={"added": {}, "changed": {}, "removed": {}},
        terminated=False,
        truncated=False,
    )


def make_trajectory(*steps: StepSnapshot) -> Trajectory:
    return Trajectory(episode_id="ep_trust_test", steps=list(steps))


# ==============================================================================
# Trust Boundary Anti-Tampering & Security Tests
# ==============================================================================

def test_trust_boundary_detects_non_monotonic_steps():
    boundary = VerifierTrustBoundary()
    # Step indices are corrupted / out of order (0, 2 instead of 0, 1)
    traj = make_trajectory(
        make_step(0, "act_1", before_hash="h0", after_hash="h1"),
        make_step(2, "act_2", before_hash="h1", after_hash="h2"),
    )
    report = boundary.evaluate(state={"key": "val"}, trajectory=traj, task=None)

    assert report.trust_level == TrustLevel.TAMPERED
    assert report.passed is False
    assert report.score == 0.0
    assert any("Non-monotonic step sequence" in e for e in report.tamper_evidence)


def test_trust_boundary_detects_state_hash_drift():
    boundary = VerifierTrustBoundary()
    # Step 0 ended at h1, but Step 1 claims to start at modified h_drift
    traj = make_trajectory(
        make_step(0, "act_1", before_hash="h0", after_hash="h1"),
        make_step(1, "act_2", before_hash="h_drift", after_hash="h2"),
    )
    report = boundary.evaluate(state={}, trajectory=traj, task=None)

    assert report.trust_level == TrustLevel.TAMPERED
    assert report.passed is False
    assert any("State hash drift" in e for e in report.tamper_evidence)


def test_trust_boundary_detects_policy_violation():
    boundary = VerifierTrustBoundary()
    step = make_step(0, "act", events=[{"type": "policy_violation", "rule": "no_egress"}])
    traj = make_trajectory(step)

    report = boundary.evaluate(state={}, trajectory=traj, task=None)
    assert report.trust_level == TrustLevel.REJECTED
    assert report.passed is False
    assert any("policy violation" in e for e in report.tamper_evidence)


def test_trust_boundary_state_defensive_deepcopy():
    boundary = VerifierTrustBoundary()
    original_state = {"nested": {"counter": 1}}

    # Adapter that attempts to mutate state
    class MutatingAdapter:
        group_type = VerifierGroup.CUSTOM

        def verify(self, spec, state, trajectory, task):
            state["nested"]["counter"] = 999
            return [CheckResult(name="check", passed=True, score=1.0)]

    boundary.register_adapter(VerifierGroup.CUSTOM, MutatingAdapter())
    spec = GroupVerificationSpec(group=VerifierGroup.CUSTOM)
    boundary.evaluate(original_state, make_trajectory(), None, spec=spec)

    # Original state remains untampered
    assert original_state["nested"]["counter"] == 1


# ==============================================================================
# CUA Adapter Tests (Computer-Use / Browser / UI)
# ==============================================================================

def test_cua_adapter_verification():
    adapter = CUAAdapter()
    spec = GroupVerificationSpec(
        group=VerifierGroup.CUA,
        rules={
            "expected_url": "/checkout/success",
            "dom_elements": {"#order-confirmation": True, ".status-badge": "Confirmed"},
        },
        milestones=["click_checkout_button", "submit_payment"],
        forbidden=["error_toast_shown"],
        expected_state={"order_id": "ORD-123"},
    )
    state = {
        "url": "https://shop.local/checkout/success",
        "dom": {"#order-confirmation": True, ".status-badge": "Confirmed"},
        "order_id": "ORD-123",
    }
    traj = make_trajectory(
        make_step(0, "click", events=[{"type": "click_checkout_button"}]),
        make_step(1, "submit", events=[{"type": "submit_payment"}]),
    )

    checks = adapter.verify(spec, state, traj, None)
    assert all(c.passed for c in checks)
    names = [c.name for c in checks]
    assert "cua:url_navigation" in names
    assert "cua:dom_element:#order-confirmation" in names
    assert "cua:milestone:submit_payment" in names
    assert "cua:forbidden:error_toast_shown" in names
    assert "cua:state:order_id" in names


def test_cua_adapter_flags_forbidden_ui_event():
    adapter = CUAAdapter()
    spec = GroupVerificationSpec(
        group=VerifierGroup.CUA,
        forbidden=["unauthorized_popup"],
    )
    traj = make_trajectory(make_step(0, "click", events=[{"type": "unauthorized_popup"}]))
    checks = adapter.verify(spec, {}, traj, None)
    forbidden_check = next(c for c in checks if c.name == "cua:forbidden:unauthorized_popup")
    assert forbidden_check.passed is False


# ==============================================================================
# TU Adapter Tests (Tool-Use / OpenAPI / APIs)
# ==============================================================================

def test_tu_adapter_verification():
    adapter = TUAdapter()
    spec = GroupVerificationSpec(
        group=VerifierGroup.TU,
        rules={
            "required_tools": ["search_users", "update_role"],
            "param_constraints": {
                "update_role": {"user_id": "usr_42", "new_role": "admin"},
            },
            "tool_sequence": ["search_users", "update_role"],
        },
        forbidden=["delete_user"],
        expected_state={"user_updated": True},
    )
    state = {"user_updated": True}
    traj = make_trajectory(
        make_step(0, "search_users", params={"query": "alice"}, before_hash="a", after_hash="b"),
        make_step(1, "update_role", params={"user_id": "usr_42", "new_role": "admin"}, before_hash="b", after_hash="c"),
    )

    checks = adapter.verify(spec, state, traj, None)
    assert all(c.passed for c in checks)
    names = [c.name for c in checks]
    assert "tu:required_tool:search_users" in names
    assert "tu:params:update_role" in names
    assert "tu:tool_sequence" in names
    assert "tu:forbidden:delete_user" in names
    assert "tu:state:user_updated" in names


def test_tu_adapter_detects_wrong_tool_sequence():
    adapter = TUAdapter()
    spec = GroupVerificationSpec(
        group=VerifierGroup.TU,
        rules={"tool_sequence": ["first_tool", "second_tool"]},
    )
    # Inverse sequence executed
    traj = make_trajectory(
        make_step(0, "second_tool"),
        make_step(1, "first_tool"),
    )
    checks = adapter.verify(spec, {}, traj, None)
    seq_check = next(c for c in checks if c.name == "tu:tool_sequence")
    assert seq_check.passed is False


# ==============================================================================
# Code Adapter Tests (Code Generation / CLI / Terminal)
# ==============================================================================

def test_code_adapter_verification():
    adapter = CodeAdapter()
    spec = GroupVerificationSpec(
        group=VerifierGroup.CODE,
        rules={
            "exit_code": 0,
            "required_files": ["src/main.py", "tests/test_main.py"],
            "forbidden_modified_files": ["conftest.py"],
        },
    )
    state = {
        "exit_code": 0,
        "tests_passed": True,
        "files": ["src/main.py", "tests/test_main.py", "README.md"],
        "modified_files": ["src/main.py"],
        "code_snippets": {
            "src/main.py": "def hello():\n    return 'world'\n",
        },
    }
    checks = adapter.verify(spec, state, make_trajectory(), None)
    assert all(c.passed for c in checks)
    names = [c.name for c in checks]
    assert "code:exit_code" in names
    assert "code:test_suite" in names
    assert "code:file_exists:src/main.py" in names
    assert "code:forbidden_file_mod:conftest.py" in names
    assert "code:syntax:src/main.py" in names


def test_code_adapter_detects_syntax_error():
    adapter = CodeAdapter()
    spec = GroupVerificationSpec(group=VerifierGroup.CODE)
    state = {
        "code_snippets": {
            "broken.py": "def broken_syntax(:\n  return\n",
        }
    }
    checks = adapter.verify(spec, state, make_trajectory(), None)
    syntax_check = next(c for c in checks if c.name == "code:syntax:broken.py")
    assert syntax_check.passed is False
    assert "Syntax error" in syntax_check.evidence


# ==============================================================================
# Data Adapter Tests (Data / Database / SQL / Analytics)
# ==============================================================================

def test_data_adapter_verification():
    adapter = DataAdapter()
    spec = GroupVerificationSpec(
        group=VerifierGroup.DATA,
        rules={
            "table_row_counts": {"orders": 5},
            "query_assertions": {"SELECT status FROM orders WHERE id=1": "shipped"},
            "required_tables": ["orders", "audit_log"],
            "forbidden_dropped_tables": ["users"],
        },
        expected_state={"migration_completed": True},
    )
    state = {
        "tables": {"orders": [1, 2, 3, 4, 5], "audit_log": [1]},
        "query_results": {"SELECT status FROM orders WHERE id=1": "shipped"},
        "dropped_tables": [],
        "migration_completed": True,
    }
    checks = adapter.verify(spec, state, make_trajectory(), None)
    assert all(c.passed for c in checks)
    names = [c.name for c in checks]
    assert "data:row_count:orders" in names
    assert "data:query_assert:SELECT status FROM orders WHERE id=1" in names
    assert "data:table_exists:orders" in names
    assert "data:forbidden_drop:users" in names
    assert "data:state:migration_completed" in names


# ==============================================================================
# VerifierEngine Integration & Dispatch Tests
# ==============================================================================

def test_verifier_engine_dispatches_group_via_trust_boundary():
    engine = VerifierEngine()

    # Task without a one-off verifier_id, declaring verifier_group="code"
    task = {
        "id": "code_task",
        "verifier_group": "code",
        "metadata": {
            "rules": {"exit_code": 0},
        },
    }
    state = {"exit_code": 0, "tests_passed": True}
    traj = make_trajectory(make_step(0, "run_tests", before_hash="h0", after_hash="h1"))

    results = engine.run_all(state, traj, task)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].score == 1.0


def test_verifier_engine_backward_compatible_with_one_off_verifiers():
    engine = VerifierEngine()
    engine.register(
        "custom_legacy",
        FunctionVerifier(lambda s, tr, t: VerificationResult.from_checks("legacy", [CheckResult(name="c", passed=True, score=1.0)])),
    )
    results = engine.run_all({}, None, {"name": "legacy_task", "verifier_id": "custom_legacy"})
    assert len(results) == 1
    assert results[0].passed is True
