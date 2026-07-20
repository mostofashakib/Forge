"""Multi-tiered per-environment verifier composition and scoring (TASKS.md #4).

``VerifierComposer`` assembles a ``LayeredVerifier`` for a task from its declared
success/failure conditions and the scenario ground truth, then scores a
verification result either binary (pass/fail) or partial (weighted per-tier).
"""

from __future__ import annotations

import pytest

from forge.envgen.agents.scenario_builder import Scenario
from forge.extraction.schemas import SuccessCondition, TaskTemplate
from forge.runtime.reward import RewardBreakdown
from forge.runtime.snapshot import StepSnapshot
from forge.runtime.trajectory import Trajectory
from forge.runtime.verification import CheckResult, VerificationResult
from forge.runtime.verifier_composer import ScoringMode, VerifierComposer


# ── Trajectory helpers ────────────────────────────────────────────────────


def _step(index: int, action_type: str, events: list[dict] | None = None) -> StepSnapshot:
    return StepSnapshot(
        episode_id="ep",
        step_index=index,
        state_hash_before="sha256:b",
        state_hash_after="sha256:a",
        action={"type": action_type},
        events=events or [],
        reward=0.0,
        verifier_results=[],
        diff={"added": {}, "changed": {}, "removed": {}},
        terminated=False,
        truncated=False,
    )


def _traj(*steps: StepSnapshot) -> Trajectory:
    return Trajectory(episode_id="ep", steps=list(steps))


def _task(success=None, failure=None, name="demo_task") -> TaskTemplate:
    return TaskTemplate(
        name=name,
        description="d",
        success_conditions=success or [],
        failure_conditions=failure or [],
    )


# ── State tier in isolation ───────────────────────────────────────────────


def test_state_tier_passes_and_fails_on_expression():
    task = _task(success=[SuccessCondition(type="state_check", expression="open_tickets == 0")])
    v = VerifierComposer().compose(task)
    assert v({"open_tickets": 0}, _traj(), {}).passed is True
    assert v({"open_tickets": 3}, _traj(), {}).passed is False


# ── Trajectory tier: ordering and forbidden actions ───────────────────────


def _ordered_scenario() -> Scenario:
    return Scenario(
        scenario_id="s",
        seed=1,
        ordering_sensitive=True,
        required_actions=["convert_lead", "archive_lead"],
        forbidden_actions=["delete_lead"],
    )


def test_required_actions_in_order_pass_wrong_order_fails():
    v = VerifierComposer().compose(_task(), scenario=_ordered_scenario())
    right = _traj(_step(0, "convert_lead"), _step(1, "archive_lead"))
    wrong = _traj(_step(0, "archive_lead"), _step(1, "convert_lead"))
    assert v({}, right, {}).passed is True
    assert v({}, wrong, {}).passed is False


def test_forbidden_action_fails_trajectory_tier():
    v = VerifierComposer().compose(_task(), scenario=_ordered_scenario())
    traj = _traj(_step(0, "convert_lead"), _step(1, "archive_lead"), _step(2, "delete_lead"))
    assert v({}, traj, {}).passed is False


def test_unordered_required_actions_accept_any_order():
    scenario = Scenario(
        scenario_id="s", seed=1, ordering_sensitive=False,
        required_actions=["a", "b"],
    )
    v = VerifierComposer().compose(_task(), scenario=scenario)
    assert v({}, _traj(_step(0, "b"), _step(1, "a")), {}).passed is True
    assert v({}, _traj(_step(0, "a")), {}).passed is False  # b missing


# ── Right answer reached the wrong way must fail ──────────────────────────


def test_correct_state_but_wrong_trajectory_order_fails():
    task = _task(success=[SuccessCondition(type="state_check", expression="done == True")])
    v = VerifierComposer().compose(task, scenario=_ordered_scenario())
    wrong = _traj(_step(0, "archive_lead"), _step(1, "convert_lead"))
    result = v({"done": True}, wrong, {})
    assert result.passed is False  # state matched, but the path was wrong


# ── Negative/side-effect tier and the false-positive guard ────────────────


def test_forbidden_side_effect_fails_even_when_state_matches():
    # The key false-positive case: final state is correct, but an unauthorized
    # side effect occurred — the verifier must NOT pass.
    task = _task(
        success=[SuccessCondition(type="state_check", expression="deleted == False")],
        failure=[SuccessCondition(type="negative_check", expression="lead_deleted")],
    )
    v = VerifierComposer().compose(task)
    traj = _traj(_step(0, "convert_lead", [{"type": "lead_deleted"}]))
    result = v({"deleted": False}, traj, {})
    assert result.passed is False


def test_no_side_effect_passes_negative_tier():
    task = _task(
        success=[SuccessCondition(type="state_check", expression="deleted == False")],
        failure=[SuccessCondition(type="negative_check", expression="lead_deleted")],
    )
    v = VerifierComposer().compose(task)
    assert v({"deleted": False}, _traj(_step(0, "convert_lead")), {}).passed is True


# ── Optional LLM-judge tier ───────────────────────────────────────────────


def test_semantic_condition_adds_judge_tier_off_by_default():
    task = _task(success=[SuccessCondition(type="semantic_check", expression="q", rubric="Was it polite?")])
    high = VerifierComposer().compose(task, judge_client=lambda rubric, s, tr, t: (0.9, "polite"))
    low = VerifierComposer().compose(task, judge_client=lambda rubric, s, tr, t: (0.2, "rude"))
    assert high({}, _traj(), {}).passed is True
    assert low({}, _traj(), {}).passed is False


def test_verifier_without_semantic_condition_needs_no_judge_client():
    task = _task(success=[SuccessCondition(type="state_check", expression="ok == True")])
    v = VerifierComposer().compose(task)  # no judge_client supplied
    assert v({"ok": True}, _traj(), {}).passed is True


# ── Scoring: binary vs partial ────────────────────────────────────────────


def _mixed_result() -> VerificationResult:
    # One tier passed, one failed → overall failed.
    return VerificationResult.from_checks(
        "v",
        [
            CheckResult(name="state:done", passed=True, score=1.0),
            CheckResult(name="trajectory:order", passed=False, score=0.0),
        ],
    )


def test_binary_scoring_gives_no_credit_to_a_failed_result():
    comp = VerifierComposer()
    rb = comp.score(_mixed_result(), ScoringMode.BINARY)
    assert isinstance(rb, RewardBreakdown)
    assert rb.total_reward == 0.0
    assert len(rb.components) == 1


def test_binary_scoring_rewards_a_fully_passing_result():
    comp = VerifierComposer()
    passing = VerificationResult.from_checks(
        "v", [CheckResult(name="state:done", passed=True, score=1.0)]
    )
    assert comp.score(passing, ScoringMode.BINARY).total_reward == 1.0


def test_partial_scoring_gives_graded_per_tier_credit():
    comp = VerifierComposer()
    rb = comp.score(_mixed_result(), ScoringMode.PARTIAL)
    # Equal-weight mean of a fully-passing state tier and a failing trajectory tier.
    assert rb.total_reward == pytest.approx(0.5)
    names = {c.name for c in rb.components}
    assert names == {"state", "trajectory"}


def test_partial_scoring_respects_per_tier_weights():
    comp = VerifierComposer()
    rb = comp.score(_mixed_result(), ScoringMode.PARTIAL, weights={"state": 3, "trajectory": 1})
    assert rb.total_reward == pytest.approx(0.75)


def test_partial_and_binary_disagree_on_a_partially_correct_result():
    comp = VerifierComposer()
    result = _mixed_result()
    assert comp.score(result, ScoringMode.BINARY).total_reward == 0.0
    assert comp.score(result, ScoringMode.PARTIAL).total_reward > 0.0


# ── Per-environment composition ───────────────────────────────────────────


def test_compose_all_builds_one_verifier_per_task():
    from forge.extraction.schemas import CompilerInput

    compiler_input = CompilerInput(
        project_name="crm",
        domain="crm",
        entities=[],
        actions=[],
        tasks=[
            _task(success=[SuccessCondition(type="state_check", expression="a == 1")], name="task_a"),
            _task(success=[SuccessCondition(type="state_check", expression="b == 1")], name="task_b"),
        ],
    )
    verifiers = VerifierComposer().compose_all(compiler_input)
    assert set(verifiers) == {"task_a", "task_b"}
    assert verifiers["task_a"]({"a": 1}, _traj(), {}).passed is True
    assert verifiers["task_b"]({"b": 0}, _traj(), {}).passed is False


# ── Live-episode integration through EnvBuilder ───────────────────────────


def test_composed_verifier_and_scoring_run_in_a_live_episode():
    import copy

    from forge.runtime.env_builder import EnvBuilder
    from forge.runtime.transition import TransitionResult

    class Factory:
        def create(self, ctx, options):
            return {"counter": {"c_0": {"id": "c_0", "value": 0}}}

    def increment(state, action, ctx):
        new = copy.deepcopy(state)
        new["counter"]["c_0"]["value"] += 1
        return TransitionResult(state=new, events=[{"type": "incremented"}])

    # One state check passes after a single increment; the other never does.
    task = _task(
        success=[
            SuccessCondition(type="state_check", expression="counter['c_0']['value'] >= 1"),
            SuccessCondition(type="state_check", expression="counter['c_0']['value'] >= 100"),
        ],
        name="reach",
    )
    scenario = Scenario(
        scenario_id="s", seed=1, ordering_sensitive=False, required_actions=["increment"]
    )

    def build(mode):
        return (
            EnvBuilder("compose_env", domain="test", max_steps=5)
            .with_initial_state(Factory())
            .with_transition("increment", increment)
            .with_composed_verifier(task, scenario=scenario)
            .with_scenario_scoring(mode)
            .with_default_task({"name": "reach", "verifier_id": "reach"})
            .build()
        )

    partial_env = build(ScoringMode.PARTIAL)
    partial_env.reset(seed=1)
    _, partial_reward, _, _, info = partial_env.step({"type": "increment"})
    # The overall verifier fails (>=100 unmet), so binary would be zero — but
    # partial scoring still credits the state tier that half-passed.
    assert info["verifier_results"][0]["passed"] is False
    assert 0.0 < partial_reward < 1.0

    binary_env = build(ScoringMode.BINARY)
    binary_env.reset(seed=1)
    _, binary_reward, _, _, _ = binary_env.step({"type": "increment"})
    assert binary_reward == 0.0
