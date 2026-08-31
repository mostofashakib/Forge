"""Regression coverage for sparse, authoritative post-episode grading."""
from __future__ import annotations

from forge.runtime.layered_verifier import LayeredVerifier

from tests.runtime.test_env import build_env


def _task(target: int = 1) -> dict:
    return {
        "name": "reach",
        "verifier_id": "check_counter",
        "inputs": {"target": target},
    }


def test_same_outcome_has_same_reward_at_different_submit_steps():
    early = build_env(max_steps=10)
    early.reset(seed=1, options={"task": _task()})
    early.step({"type": "increment"})
    _, early_reward, _, _, _ = early.step({"type": "submit"})

    late = build_env(max_steps=10)
    late.reset(seed=1, options={"task": _task()})
    for _ in range(4):
        late.step({"type": "increment"})
    _, late_reward, _, _, _ = late.step({"type": "submit"})

    assert early_reward == late_reward == 1.0


def test_llm_judge_runs_exactly_once_for_a_multi_step_episode():
    calls = 0

    def judge(*_args):
        nonlocal calls
        calls += 1
        return 1.0, "complete"

    env = build_env(max_steps=10)
    verifier = LayeredVerifier("judge", judge_client=judge)
    verifier.add_llm_judge("quality", "Is the task complete?")
    env._verifier_engine.register("judge", verifier)
    task = {"name": "judged", "verifier_id": "judge", "inputs": {}}
    env.reset(seed=1, options={"task": task})
    env.step({"type": "increment"})
    env.step({"type": "increment"})
    assert calls == 0

    _, reward, terminated, _, info = env.step({"type": "submit"})

    assert terminated is True
    assert reward == 1.0
    assert info["passed"] is True
    assert calls == 1


def test_submit_terminates_even_when_final_verdict_fails():
    env = build_env()
    env.reset(seed=1, options={"task": _task(target=99)})

    _, reward, terminated, truncated, info = env.step({"type": "submit"})

    assert terminated is True and truncated is False
    assert reward == 0.0
    assert info["passed"] is False
    assert info["termination_reason"] == "submitted"


def test_invalid_actions_still_reach_the_step_budget_and_finalize():
    env = build_env(max_steps=2)
    env.reset(seed=1, options={"task": _task(target=99)})
    env.step({"type": "bad"})

    _, reward, terminated, truncated, info = env.step({"type": "still_bad"})

    assert terminated is False and truncated is True
    assert reward == 0.0
    assert info["termination_reason"] == "max_steps"


def test_finalization_is_idempotent_after_budget_exhaustion():
    env = build_env(max_steps=1)
    env.reset(seed=1, options={"task": _task(target=99)})
    env.step({"type": "increment"})

    first = env.finalize_episode()
    second = env.finalize_episode()

    assert first is second
