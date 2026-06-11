# tests/runtime/test_layered_verifier.py
import pytest
from forge.runtime.layered_verifier import LayeredVerifier
from forge.runtime.snapshot import StepSnapshot
from forge.runtime.trajectory import Trajectory
from forge.runtime.verification import VerificationResult
from forge.runtime.verifier import VerifierEngine


def make_step(index: int, action_type: str, events: list[dict] | None = None) -> StepSnapshot:
    return StepSnapshot(
        episode_id="ep_test",
        step_index=index,
        state_hash_before="sha256:before",
        state_hash_after="sha256:after",
        action={"type": action_type},
        events=events or [],
        reward=0.0,
        verifier_results=[],
        diff={"added": {}, "changed": {}, "removed": {}},
        terminated=False,
        truncated=False,
    )


def make_trajectory(*steps: StepSnapshot) -> Trajectory:
    return Trajectory(episode_id="ep_test", steps=list(steps))


STATE = {"emails": {"e_0": {"id": "e_0", "replied": True, "deleted": False}}}
TASK = {"name": "reply_task", "verifier_id": "reply_task", "inputs": {"email_id": "e_0"}}


def test_all_layers_pass():
    v = LayeredVerifier("reply_task")
    v.add_state_check("email_replied", lambda s, tr, t: s["emails"]["e_0"]["replied"])
    v.require_milestones(["email_opened", "email_replied"])
    v.require_actions(["open_email", "reply_email"])
    v.add_negative_check("nothing_deleted", lambda s, tr, t: not s["emails"]["e_0"]["deleted"])

    traj = make_trajectory(
        make_step(0, "open_email", [{"type": "email_opened"}]),
        make_step(1, "reply_email", [{"type": "email_replied"}]),
    )
    result = v(STATE, traj, TASK)
    assert isinstance(result, VerificationResult)
    assert result.passed is True
    assert result.score == 1.0


def test_check_names_carry_layer_prefix():
    v = LayeredVerifier("reply_task")
    v.add_state_check("email_replied", lambda s, tr, t: True)
    v.add_invariant_check("milestone", lambda s, tr, t: True)
    v.add_trajectory_check("tool_calls", lambda s, tr, t: True)
    v.add_negative_check("no_deletes", lambda s, tr, t: True)

    result = v(STATE, make_trajectory(), TASK)
    names = [c.name for c in result.checks]
    assert names == [
        "state:email_replied",
        "invariant:milestone",
        "trajectory:tool_calls",
        "negative:no_deletes",
    ]


def test_failing_state_check_fails_result():
    v = LayeredVerifier("reply_task")
    v.add_state_check("email_replied", lambda s, tr, t: False)
    result = v(STATE, make_trajectory(), TASK)
    assert result.passed is False


def test_milestones_out_of_order_fail_invariant_layer():
    v = LayeredVerifier("reply_task")
    v.require_milestones(["email_opened", "email_replied"])
    traj = make_trajectory(
        make_step(0, "reply_email", [{"type": "email_replied"}]),
        make_step(1, "open_email", [{"type": "email_opened"}]),
    )
    result = v(STATE, traj, TASK)
    assert result.passed is False
    failed = [c for c in result.checks if not c.passed]
    assert failed and failed[0].name.startswith("invariant:")


def test_missing_required_action_fails_trajectory_layer():
    v = LayeredVerifier("reply_task")
    v.require_actions(["reply_email"])
    result = v(STATE, make_trajectory(make_step(0, "open_email")), TASK)
    assert result.passed is False


def test_forbidden_action_fails_trajectory_layer():
    v = LayeredVerifier("reply_task")
    v.forbid_actions(["delete_email"])
    traj = make_trajectory(make_step(0, "reply_email"), make_step(1, "delete_email"))
    result = v(STATE, traj, TASK)
    assert result.passed is False
    failed = [c for c in result.checks if not c.passed]
    assert "delete_email" in (failed[0].evidence or "")


def test_forbidden_event_fails_negative_layer():
    v = LayeredVerifier("reply_task")
    v.forbid_events(["email_deleted"])
    traj = make_trajectory(make_step(0, "cleanup", [{"type": "email_deleted"}]))
    result = v(STATE, traj, TASK)
    assert result.passed is False
    failed = [c for c in result.checks if not c.passed]
    assert failed[0].name.startswith("negative:")


def test_llm_judge_uses_client_and_threshold():
    def fake_client(rubric, state, trajectory, task):
        return 0.9, "well-written reply"

    v = LayeredVerifier("reply_task", judge_client=fake_client)
    v.add_llm_judge("reply_quality", rubric="Reply must be polite and address the question")
    result = v(STATE, make_trajectory(), TASK)
    assert result.passed is True
    judge_check = result.checks[0]
    assert judge_check.name == "judge:reply_quality"
    assert judge_check.score == 0.9
    assert "well-written" in judge_check.evidence


def test_llm_judge_below_threshold_fails():
    v = LayeredVerifier("reply_task", judge_client=lambda r, s, tr, t: (0.3, "off-topic"))
    v.add_llm_judge("reply_quality", rubric="rubric", threshold=0.7)
    result = v(STATE, make_trajectory(), TASK)
    assert result.passed is False


def test_llm_judge_without_client_raises():
    v = LayeredVerifier("reply_task")
    v.add_llm_judge("reply_quality", rubric="rubric")
    with pytest.raises(RuntimeError, match="judge"):
        v(STATE, make_trajectory(), TASK)


def test_check_can_return_evidence_tuple():
    v = LayeredVerifier("reply_task")
    v.add_state_check("email_replied", lambda s, tr, t: (False, "replied flag is False"))
    result = v(STATE, make_trajectory(), TASK)
    assert result.checks[0].evidence == "replied flag is False"


def test_registers_with_verifier_engine():
    v = LayeredVerifier("reply_task")
    v.add_state_check("email_replied", lambda s, tr, t: s["emails"]["e_0"]["replied"])

    engine = VerifierEngine()
    engine.register("reply_task", v)
    results = engine.run_all(STATE, make_trajectory(), TASK)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].verifier_id == "reply_task"
