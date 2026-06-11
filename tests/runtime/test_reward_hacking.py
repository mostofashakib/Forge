# tests/runtime/test_reward_hacking.py
from forge.runtime.reward_hacking import AuditReport, RewardHackingAuditor
from forge.runtime.snapshot import StepSnapshot
from forge.runtime.trajectory import Trajectory
from forge.runtime.verification import CheckResult, VerificationResult


def make_step(index: int, action: dict, events: list[dict] | None = None, reward: float = 0.0) -> StepSnapshot:
    return StepSnapshot(
        episode_id="ep_test",
        step_index=index,
        state_hash_before="sha256:before",
        state_hash_after="sha256:after",
        action=action,
        events=events or [],
        reward=reward,
        verifier_results=[],
        diff={"added": {}, "changed": {}, "removed": {}},
        terminated=False,
        truncated=False,
    )


def make_trajectory(*steps: StepSnapshot) -> Trajectory:
    return Trajectory(episode_id="ep_test", steps=list(steps))


def passed_verification() -> VerificationResult:
    return VerificationResult.from_checks(
        "task", [CheckResult(name="state:done", passed=True, score=1.0)]
    )


def failed_verification() -> VerificationResult:
    return VerificationResult.from_checks(
        "task", [CheckResult(name="state:done", passed=False, score=0.0)]
    )


TASK = {"name": "task", "verifier_id": "task"}


def test_clean_episode_is_not_flagged():
    auditor = RewardHackingAuditor(required_milestones=["opened", "replied"], min_steps=2)
    traj = make_trajectory(
        make_step(0, {"type": "open"}, [{"type": "opened"}]),
        make_step(1, {"type": "reply"}, [{"type": "replied"}], reward=1.0),
    )
    report = auditor.audit({}, traj, TASK, passed_verification())
    assert isinstance(report, AuditReport)
    assert report.flagged is False
    assert report.findings == []


def test_passed_with_skipped_milestones_is_flagged():
    auditor = RewardHackingAuditor(required_milestones=["opened", "replied"])
    traj = make_trajectory(make_step(0, {"type": "reply"}, [{"type": "replied"}], reward=1.0))
    report = auditor.audit({}, traj, TASK, passed_verification())
    assert report.flagged is True
    codes = [f.code for f in report.findings]
    assert "milestones_skipped" in codes
    finding = next(f for f in report.findings if f.code == "milestones_skipped")
    assert "opened" in finding.detail
    assert finding.severity == "high"


def test_failed_verification_skips_milestone_audit():
    auditor = RewardHackingAuditor(required_milestones=["opened", "replied"])
    traj = make_trajectory(make_step(0, {"type": "noop"}))
    report = auditor.audit({}, traj, TASK, failed_verification())
    assert "milestones_skipped" not in [f.code for f in report.findings]


def test_suspiciously_short_passing_episode_is_flagged():
    auditor = RewardHackingAuditor(min_steps=3)
    traj = make_trajectory(make_step(0, {"type": "win"}, reward=1.0))
    report = auditor.audit({}, traj, TASK, passed_verification())
    assert "suspiciously_short" in [f.code for f in report.findings]


def test_redundant_repeated_actions_are_flagged():
    auditor = RewardHackingAuditor(max_action_repeats=2)
    same = {"type": "refresh", "target": "inbox"}
    traj = make_trajectory(
        make_step(0, dict(same)),
        make_step(1, dict(same)),
        make_step(2, dict(same)),
        make_step(3, {"type": "reply"}),
    )
    report = auditor.audit({}, traj, TASK, failed_verification())
    assert "redundant_actions" in [f.code for f in report.findings]


def test_llm_audit_client_finding_is_included():
    def fake_client(state, trajectory, task, verification):
        return True, "agent wrote the answer directly into state without doing the task"

    auditor = RewardHackingAuditor(llm_client=fake_client)
    traj = make_trajectory(make_step(0, {"type": "win"}, reward=1.0))
    report = auditor.audit({}, traj, TASK, passed_verification())
    assert report.flagged is True
    llm_finding = next(f for f in report.findings if f.code == "llm_audit")
    assert "directly into state" in llm_finding.detail


def test_llm_audit_client_clean_verdict_adds_nothing():
    auditor = RewardHackingAuditor(llm_client=lambda s, tr, t, v: (False, "looks legitimate"))
    traj = make_trajectory(
        make_step(0, {"type": "open"}),
        make_step(1, {"type": "reply"}),
    )
    report = auditor.audit({}, traj, TASK, passed_verification())
    assert report.flagged is False


def test_auditor_pairs_with_layered_verifier_milestones():
    from forge.runtime.layered_verifier import LayeredVerifier

    verifier = LayeredVerifier("task")
    verifier.add_state_check("done", lambda s, tr, t: True)
    verifier.require_milestones(["opened", "replied"])
    auditor = RewardHackingAuditor.for_verifier(verifier)

    traj = make_trajectory(make_step(0, {"type": "reply"}, [{"type": "replied"}]))
    verification = verifier({}, traj, TASK)
    report = auditor.audit({}, traj, TASK, verification)
    # The verifier itself fails the invariant layer; the auditor inherits the
    # milestone list and reports nothing extra for an already-failed episode.
    assert verification.passed is False
    assert "milestones_skipped" not in [f.code for f in report.findings]

    full_traj = make_trajectory(
        make_step(0, {"type": "open"}, [{"type": "opened"}]),
        make_step(1, {"type": "reply"}, [{"type": "replied"}]),
    )
    verification = verifier({}, full_traj, TASK)
    report = auditor.audit({}, full_traj, TASK, verification)
    assert verification.passed is True
    assert report.flagged is False
