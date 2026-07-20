"""End-to-end determinism/seed-control test for the container contract.

Drives a faithful reference implementation of the seeded ``/forge/reset``
contract through the real :class:`ContainerEnvBase` plumbing to prove the three
required properties:

  1. every rollout starts from a known position and resets fresh;
  2. the same seed always produces the same starting state (and a different seed
     a different-but-reproducible one);
  3. the same starting state plus the same actions always produces the same
     result.
"""
from __future__ import annotations

import json
import random

import httpx

from forge.envgen.container_env_base import ContainerEnvBase


def _seeded_app_handler():
    """A minimal, deterministic app honoring the seeded reset contract.

    Mirrors the generated-app rules: a module-like state, a logical clock and
    id counters both reset on (re)seed, and all variation drawn from
    ``random.Random(seed)`` so identical seeds reproduce and distinct seeds
    diverge.
    """
    state = {"clock": 0, "counters": {}, "items": []}

    def next_id(entity: str) -> int:
        state["counters"][entity] = state["counters"].get(entity, 0) + 1
        return state["counters"][entity]

    def seed_state(seed: int) -> None:
        state["clock"] = 0
        state["counters"].clear()
        state["items"].clear()
        rng = random.Random(seed)
        for _ in range(rng.randint(1, 5)):
            state["items"].append({
                "id": next_id("item"),
                "value": rng.randint(0, 100),
                "created_at": state["clock"],
            })
            state["clock"] += 1

    def reset_state() -> None:
        seed_state(0)  # fixed baseline

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/forge/reset":
            seed = None
            if request.content:
                seed = json.loads(request.content).get("seed")
            if seed is not None:
                seed_state(seed)
            else:
                reset_state()
            return httpx.Response(200, json={"ok": True})
        if path == "/forge/state":
            return httpx.Response(200, json={"items": list(state["items"])})
        if path == "/append":
            state["items"].append({
                "id": next_id("item"), "value": -1, "created_at": state["clock"],
            })
            state["clock"] += 1
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"error": "unknown"})

    return handler


def _env() -> ContainerEnvBase:
    client = httpx.Client(
        transport=httpx.MockTransport(_seeded_app_handler()), base_url="http://app"
    )
    return ContainerEnvBase("http://app", client=client)


# 1. Fresh, known start ------------------------------------------------------

def test_unseeded_reset_is_a_fixed_reproducible_baseline():
    env = _env()
    first, _ = env.reset()
    second, _ = env.reset()
    assert first == second                       # every rollout starts fresh
    baseline_via_seed_zero, _ = env.reset(seed=0)
    assert first == baseline_via_seed_zero       # baseline == seed 0


# 2. Same seed → same start; different seed → different start ----------------

def test_same_seed_reproduces_starting_state():
    env = _env()
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert a == b


def test_different_seeds_produce_different_starting_states():
    env = _env()
    a, _ = env.reset(seed=1)
    b, _ = env.reset(seed=2)
    assert a != b


# 3. Same start + same actions → same result --------------------------------

def test_same_seed_plus_same_actions_produce_same_result():
    env1, env2 = _env(), _env()
    env1.reset(seed=42)
    env2.reset(seed=42)
    for _ in range(3):
        env1.step({"type": "append"})
        env2.step({"type": "append"})
    assert env1._observe() == env2._observe()


def test_diverging_actions_produce_diverging_results():
    # Guard against a false positive: identical seed but different action counts
    # must NOT land in the same state.
    env1, env2 = _env(), _env()
    env1.reset(seed=42)
    env2.reset(seed=42)
    env1.step({"type": "append"})
    assert env1._observe() != env2._observe()
