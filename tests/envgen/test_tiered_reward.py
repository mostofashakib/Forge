"""Tests for the tiered CLI reward engine.

Three tiers, three concerns:
  1. plan_end_state — LLM converts an objective into assertions + step estimate
  2. LoopDetector  — kills stuck/looping agents (no LLM cost, pure heuristic)
  3. grade         — runs assertions, applies efficiency, falls back to LLM
                     partial credit only when no assertion passed
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from forge.envgen.tiered_reward import (
    EndStateSpec,
    LoopDetector,
    LoopDetectorConfig,
    TieredRewardConfig,
    TieredRewardEngine,
)


# ---------------------------------------------------------------------------
# LoopDetector — pure-heuristic, no LLM
# ---------------------------------------------------------------------------

def test_loop_detector_quiet_run_returns_none():
    """A diverse trajectory of unique commands should never fire."""
    d = LoopDetector(LoopDetectorConfig(repeat_threshold=3, consecutive_failure_threshold=5))
    for i in range(8):
        assert d.observe(f"echo {i}", exit_code=0, stdout=f"out{i}") is None


def test_loop_detector_fires_on_repeated_command_and_output():
    d = LoopDetector(LoopDetectorConfig(repeat_threshold=3, consecutive_failure_threshold=99))
    assert d.observe("ls /nope", 1, "no such file") is None
    assert d.observe("ls /nope", 1, "no such file") is None
    # Third identical fingerprint should trigger.
    assert d.observe("ls /nope", 1, "no such file") == "loop_detected"


def test_loop_detector_does_not_fire_when_output_changes():
    """Same command with different output isn't a loop — agent is making progress."""
    d = LoopDetector(LoopDetectorConfig(repeat_threshold=3, consecutive_failure_threshold=99))
    assert d.observe("ls /tmp", 0, "a") is None
    assert d.observe("ls /tmp", 0, "a b") is None
    assert d.observe("ls /tmp", 0, "a b c") is None


def test_loop_detector_fires_on_consecutive_failures():
    d = LoopDetector(LoopDetectorConfig(repeat_threshold=99, consecutive_failure_threshold=4))
    assert d.observe("cmd1", 1, "err1") is None
    assert d.observe("cmd2", 2, "err2") is None
    assert d.observe("cmd3", 1, "err3") is None
    # Fourth consecutive failure — agent thrashing.
    assert d.observe("cmd4", 1, "err4") == "stuck_failing"


def test_loop_detector_resets_failure_streak_on_success():
    d = LoopDetector(LoopDetectorConfig(repeat_threshold=99, consecutive_failure_threshold=3))
    d.observe("a", 1, "fail")
    d.observe("b", 1, "fail")
    d.observe("ok", 0, "good")  # reset
    # Two more failures shouldn't trigger — streak was reset.
    assert d.observe("c", 1, "fail") is None
    assert d.observe("d", 1, "fail") is None


# ---------------------------------------------------------------------------
# plan_end_state — LLM-driven spec generation (with fallback)
# ---------------------------------------------------------------------------

def test_plan_end_state_returns_llm_spec():
    """Happy path: LLM returns valid spec, engine surfaces it."""
    from forge.envgen.tiered_reward import _EndStateSpecLLM, _SpecAssertion
    mock_client = MagicMock()
    mock_client.extract.return_value = _EndStateSpecLLM(
        summary="A file /tmp/foo.txt exists with content 'hello'.",
        expected_steps=2,
        assertions=[
            _SpecAssertion(description="file exists", command="[[ -f /tmp/foo.txt ]]"),
            _SpecAssertion(description="content matches", command='[[ "$(cat /tmp/foo.txt)" == "hello" ]]'),
        ],
    )
    engine = TieredRewardEngine(client=mock_client)
    spec = engine.plan_end_state("create /tmp/foo.txt with content 'hello'")
    assert spec.expected_steps == 2
    assert len(spec.assertions) == 2
    assert spec.assertions[0]["command"] == "[[ -f /tmp/foo.txt ]]"


def test_plan_end_state_falls_back_when_llm_fails():
    """LLM error must not crash episode start — return a minimal spec."""
    mock_client = MagicMock()
    mock_client.extract.side_effect = RuntimeError("network down")
    engine = TieredRewardEngine(client=mock_client)
    spec = engine.plan_end_state("create a file")
    assert spec.summary == "create a file"
    assert spec.assertions == []
    assert spec.expected_steps >= 1


# ---------------------------------------------------------------------------
# grade — Tier 3
# ---------------------------------------------------------------------------

def _make_engine(mock_client=None) -> TieredRewardEngine:
    return TieredRewardEngine(
        client=mock_client or MagicMock(),
        config=TieredRewardConfig(min_efficiency=0.5, llm_grade_when_zero_pass=True),
    )


def test_grade_all_assertions_pass_full_reward_for_optimal_run():
    """All assertions pass + agent took ≤ expected_steps → reward = 1.0."""
    engine = _make_engine()
    spec = EndStateSpec(
        summary="x", expected_steps=5,
        assertions=[
            {"description": "a1", "command": "true"},
            {"description": "a2", "command": "true"},
        ],
    )
    history = [{"command": "c", "exit_code": 0, "stdout": ""} for _ in range(5)]

    with patch("forge.envgen.tiered_reward.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        grade = engine.grade("obj", spec, history, container_id="cid")

    assert grade.test_pass_rate == 1.0
    assert grade.efficiency_factor == 1.0
    assert grade.final_reward == pytest.approx(1.0)
    assert grade.early_termination is None


def test_grade_partial_pass_yields_proportional_reward():
    """6/10 assertions pass → reward ≈ efficiency × 0.6."""
    engine = _make_engine()
    spec = EndStateSpec(
        summary="x", expected_steps=10,
        assertions=[{"description": f"a{i}", "command": f"check-{i}"} for i in range(10)],
    )
    history = [{"command": "c", "exit_code": 0, "stdout": ""} for _ in range(10)]

    # 6 assertions pass, 4 fail
    side_effects = [MagicMock(returncode=0 if i < 6 else 1, stdout="", stderr="")
                    for i in range(10)]
    with patch("forge.envgen.tiered_reward.subprocess.run", side_effect=side_effects):
        grade = engine.grade("obj", spec, history, container_id="cid")

    assert grade.test_pass_rate == pytest.approx(0.6)
    assert grade.efficiency_factor == 1.0
    assert grade.final_reward == pytest.approx(0.6)
    # LLM should NOT have been consulted — at least one assertion passed.
    assert grade.partial_credit == 0.0


def test_grade_efficiency_penalty_for_taking_too_many_steps():
    """expected_steps=5, actual=20 → efficiency = 5/20 = 0.25, clamped to 0.5."""
    engine = _make_engine()
    spec = EndStateSpec(summary="x", expected_steps=5, assertions=[
        {"description": "a", "command": "true"},
    ])
    history = [{"command": "c", "exit_code": 0, "stdout": ""} for _ in range(20)]

    with patch("forge.envgen.tiered_reward.subprocess.run",
               return_value=MagicMock(returncode=0, stdout="", stderr="")):
        grade = engine.grade("obj", spec, history, container_id="cid")

    # All assertions passed but the agent was 4× over budget.
    assert grade.test_pass_rate == 1.0
    assert grade.efficiency_factor == 0.5  # min_efficiency floor
    assert grade.final_reward == pytest.approx(0.5)


def test_grade_loop_detected_zeroes_reward_immediately():
    """Tier 2 killed the episode → reward = 0, no assertions run, no LLM call."""
    engine = _make_engine()
    spec = EndStateSpec(summary="x", expected_steps=5, assertions=[
        {"description": "a", "command": "true"},
    ])
    history = [{"command": "loop", "exit_code": 0, "stdout": "x"} for _ in range(7)]

    with patch("forge.envgen.tiered_reward.subprocess.run") as mock_run:
        grade = engine.grade(
            "obj", spec, history, container_id="cid",
            early_termination="loop_detected",
        )

    assert grade.final_reward == 0.0
    assert grade.early_termination == "loop_detected"
    mock_run.assert_not_called()  # short-circuit before running assertions


def test_grade_zero_pass_invokes_llm_partial_credit():
    """No assertions passed → fall through to LLM grader, get partial credit."""
    from forge.envgen.tiered_reward import _PartialCreditLLM
    mock_client = MagicMock()
    mock_client.extract.return_value = _PartialCreditLLM(
        score=0.25, reasoning="Got the right tool but wrong file path.",
    )
    engine = TieredRewardEngine(client=mock_client)
    spec = EndStateSpec(
        summary="x", expected_steps=5,
        assertions=[
            {"description": "a", "command": "fail-1"},
            {"description": "b", "command": "fail-2"},
        ],
    )
    history = [{"command": "c", "exit_code": 0, "stdout": ""} for _ in range(5)]

    with patch("forge.envgen.tiered_reward.subprocess.run",
               return_value=MagicMock(returncode=1, stdout="", stderr="")):
        grade = engine.grade("obj", spec, history, container_id="cid")

    assert grade.test_pass_rate == 0.0
    assert grade.partial_credit == pytest.approx(0.25)
    assert grade.final_reward == pytest.approx(0.25)  # efficiency=1.0 here


def test_grade_caps_partial_credit_at_0_4():
    """Even if the LLM grader returns 0.9, partial credit is capped at 0.4
    (Pydantic schema enforces it). Successful trajectories should pass tests."""
    from forge.envgen.tiered_reward import _PartialCreditLLM
    mock_client = MagicMock()
    # Return-value validates against the schema; le=0.4 is enforced.
    with pytest.raises(Exception):
        _PartialCreditLLM(score=0.9, reasoning="too generous")


def test_grade_partial_credit_llm_failure_falls_back_to_zero():
    """If the LLM call itself errors out, partial credit is 0 (not a crash)."""
    mock_client = MagicMock()
    mock_client.extract.side_effect = RuntimeError("LLM down")
    engine = TieredRewardEngine(client=mock_client)
    spec = EndStateSpec(summary="x", expected_steps=5, assertions=[
        {"description": "a", "command": "fail"},
    ])
    history = [{"command": "c", "exit_code": 0, "stdout": ""}]

    with patch("forge.envgen.tiered_reward.subprocess.run",
               return_value=MagicMock(returncode=1, stdout="", stderr="")):
        grade = engine.grade("obj", spec, history, container_id="cid")

    assert grade.partial_credit == 0.0
    assert grade.final_reward == 0.0


def test_grade_handles_assertion_timeout_gracefully():
    """A hung assertion shouldn't crash the grader — it counts as failed."""
    engine = _make_engine()
    spec = EndStateSpec(summary="x", expected_steps=5, assertions=[
        {"description": "slow", "command": "sleep 9999"},
        {"description": "fast", "command": "true"},
    ])
    history = [{"command": "c", "exit_code": 0, "stdout": ""} for _ in range(5)]

    side_effects = [
        subprocess.TimeoutExpired(cmd="docker exec", timeout=15),
        MagicMock(returncode=0, stdout="", stderr=""),
    ]
    with patch("forge.envgen.tiered_reward.subprocess.run", side_effect=side_effects):
        grade = engine.grade("obj", spec, history, container_id="cid")

    assert grade.test_pass_rate == 0.5
    assert grade.test_results[0].passed is False
    assert "timed out" in grade.test_results[0].stderr
    assert grade.test_results[1].passed is True


def test_grade_no_assertions_in_spec_uses_llm_only():
    """When the planner LLM failed and assertions list is empty, the engine
    must still produce a grade (via partial-credit LLM)."""
    from forge.envgen.tiered_reward import _PartialCreditLLM
    mock_client = MagicMock()
    mock_client.extract.return_value = _PartialCreditLLM(score=0.1, reasoning="x")
    engine = TieredRewardEngine(client=mock_client)
    spec = EndStateSpec(summary="x", expected_steps=5, assertions=[])
    history = [{"command": "c", "exit_code": 0, "stdout": ""}]

    with patch("forge.envgen.tiered_reward.subprocess.run") as mock_run:
        grade = engine.grade("obj", spec, history, container_id="cid")
        # No assertions to run.
        mock_run.assert_not_called()

    assert grade.partial_credit == pytest.approx(0.1)
    assert grade.final_reward == pytest.approx(0.1)
