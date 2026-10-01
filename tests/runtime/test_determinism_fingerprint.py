"""Unit tests for environment fingerprint check in determinism testing (Feature Request 6)."""
from __future__ import annotations

import pytest

from forge.runtime.determinism import (
    DeterminismError,
    DeterminismReport,
    check_fingerprint_determinism,
    run_determinism_check,
)
from forge.runtime.env import ForgeEnv
from forge.runtime.snapshot import EnvironmentSpec
from forge.runtime.transition import (
    FunctionTransitionHandler,
    TransitionEngine,
    TransitionResult,
)
from forge.runtime.verifier import VerifierEngine
from forge.runtime.reward import RewardEngine


class _InitialState:
    def reset(self, ctx, seed=None, options=None):
        actual_seed = seed if seed is not None else 0
        task_id = (options or {}).get("task", "default")
        return {"counter": 0, "seed": actual_seed, "task": str(task_id)}


def _make_test_env():
    spec = EnvironmentSpec(name="test_fp_env", domain="test", max_steps=10)
    return ForgeEnv(
        env_spec=spec,
        initial_state_provider=_InitialState(),
        transition_engine=TransitionEngine(),
        verifier_engine=VerifierEngine(),
        reward_engine=RewardEngine(),
    )


def test_environments_expose_and_store_fingerprint():
    env = _make_test_env()
    obs, info = env.reset(seed=42)
    assert "fingerprint" in info
    assert len(info["fingerprint"]) == 64
    assert env.fingerprint() == info["fingerprint"]


def test_fingerprint_matches_between_initial_creation_and_snapshot_recreate():
    env = _make_test_env()

    # Recreate function simulates recreating from snapshot for the same (task, seed)
    def recreate_fn(task=None, seed=None):
        new_env = _make_test_env()
        new_env.reset(seed=seed, options={"task": task} if task else None)
        return new_env

    initial_fp, recreated_fp, ok = check_fingerprint_determinism(
        env, task="task_A", seed=10, recreate_fn=recreate_fn
    )
    assert ok is True
    assert initial_fp == recreated_fp
    assert len(initial_fp) == 64


def test_determinism_check_verifies_fingerprint():
    env = _make_test_env()

    def recreate_fn(task=None, seed=None):
        new_env = _make_test_env()
        new_env.reset(seed=seed, options={"task": task} if task else None)
        return new_env

    report = run_determinism_check(
        env, seed=7, num_steps=2, task="classify_email", recreate_fn=recreate_fn
    )
    assert isinstance(report, DeterminismReport)
    assert report.passed is True
    assert report.fingerprint_matched is True
    assert report.fingerprint == report.recreated_fingerprint
    assert report.task == "classify_email"


def test_fingerprint_mismatch_raises_determinism_error():
    env = _make_test_env()

    # Broken recreate function that produces a different state / fingerprint
    def bad_recreate_fn(task=None, seed=None):
        new_env = _make_test_env()
        new_env.reset(seed=(seed or 0) + 999, options={"task": "diverged"} if task else None)
        return new_env

    with pytest.raises(DeterminismError) as exc_info:
        run_determinism_check(
            env, seed=12, num_steps=2, task="sort_items", recreate_fn=bad_recreate_fn
        )

    assert "fingerprint mismatch" in str(exc_info.value).lower()
