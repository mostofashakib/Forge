"""The shipped example, exercised as an environment.

Worth testing rather than just shipping: it is the reference an author copies,
and an example that stopped working would teach the wrong thing quietly.
"""
from __future__ import annotations

import pytest

from examples.clinical_handoff.env import build_ward_env


def run(env, steps=12):
    env.reset(seed=3)
    for _ in range(steps):
        obs, _, terminated, truncated, info = env.step(
            {"type": "send_message", "to": "supervisor", "body": "ready, notes attached"}
        )
        if terminated or truncated:
            break
    return obs, info


def test_the_cast_is_present_on_reset():
    env = build_ward_env()
    _obs, info = env.reset(seed=3)
    assert [p["id"] for p in info["personas"]] == ["supervisor", "nurse"]


def test_the_task_is_unreachable_without_the_cast():
    """The point of the example: the agent cannot approve its own discharge."""
    env = build_ward_env(with_personas=False)
    obs, _ = run(env)
    assert obs["patients"]["pt_0000"]["discharge_approved"] == 0


def test_the_supervisor_eventually_approves_the_discharge():
    env = build_ward_env()
    obs, _ = run(env)
    assert obs["patients"]["pt_0000"]["discharge_approved"] == 1


def test_the_nurse_cannot_approve_a_discharge():
    """Only the supervisor holds that action; the guard is what enforces it."""
    env = build_ward_env()
    env.reset(seed=3)
    approvals = []
    for _ in range(12):
        _obs, _, terminated, truncated, info = env.step(
            {"type": "send_message", "to": "supervisor", "body": "ready"}
        )
        for turn in info.get("persona_turns", []):
            action = turn.get("action") or {}
            if action.get("type") == "approve_discharge":
                approvals.append(turn["persona_id"])
        if terminated or truncated:
            break
    assert approvals
    assert set(approvals) == {"supervisor"}


def test_the_supervisor_does_not_answer_on_the_step_they_were_asked():
    """Latency is what makes waiting for a colleague part of the task."""
    env = build_ward_env()
    env.reset(seed=3)
    _obs, _, _, _, info = env.step(
        {"type": "send_message", "to": "supervisor", "body": "ready"}
    )
    acted = {t["persona_id"] for t in info.get("persona_turns", [])}
    assert "supervisor" not in acted


def test_the_episode_is_reproducible_for_a_seed():
    def transcript():
        env = build_ward_env()
        env.reset(seed=3)
        out = []
        for _ in range(8):
            _obs, _, terminated, truncated, info = env.step(
                {"type": "send_message", "to": "supervisor", "body": "ready"}
            )
            out.append(
                [(t["persona_id"], (t.get("action") or {}).get("type")) for t in info.get("persona_turns", [])]
            )
            if terminated or truncated:
                break
        return out

    assert transcript() == transcript()


def test_the_generated_config_stub_parses():
    """The `personas:` block new environments ship with must stay loadable."""
    import yaml

    from forge.compiler.package_builder import _CUSTOM_STUBS
    from forge.personas.config import load_population

    raw = yaml.safe_load(_CUSTOM_STUBS["config.yaml"])
    assert load_population(raw["personas"]).enabled is False


def test_the_generated_config_stub_documents_the_action_guardrail():
    from forge.compiler.package_builder import _CUSTOM_STUBS

    assert "allowed_actions" in _CUSTOM_STUBS["config.yaml"]
