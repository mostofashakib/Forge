from __future__ import annotations
import hashlib
import json
import random
from dataclasses import dataclass

from forge.runtime.errors import DeterminismError

__all__ = ["DeterminismError", "DeterminismReport", "run_determinism_check"]


@dataclass
class DeterminismReport:
    passed: bool
    seed: int
    observation_hash: str
    actions: list[dict]
    total_reward: float


def _canonical_hash(obs: dict) -> str:
    return hashlib.sha256(
        json.dumps(obs, sort_keys=True, default=str).encode()
    ).hexdigest()


def _rollout(
    env, seed: int, num_steps: int, actions: list[dict] | None
) -> tuple[list[str], list[dict], float]:
    """Reset with `seed`, run up to `num_steps`, return per-step hashes, actions
    taken, and the total reward.

    Each step's hash covers the observation, reward, and termination flags, so
    a divergent score fails the check just like a divergent observation. If
    `actions` is None, actions are generated from a RNG seeded with `seed` over
    the env's registered action types, so both rollouts can regenerate or
    replay the identical sequence.
    """
    obs, _info = env.reset(seed=seed)
    step_hashes = [_canonical_hash(obs)]
    taken: list[dict] = []
    total_reward = 0.0

    if actions is None:
        action_types = sorted(env.action_types)
        if not action_types:
            return step_hashes, taken, total_reward
        rng = random.Random(seed)
        planned = [{"type": rng.choice(action_types)} for _ in range(num_steps)]
    else:
        planned = actions[:num_steps]

    for action in planned:
        try:
            obs, reward, terminated, truncated, _info = env.step(action)
        except Exception as exc:
            # A transition rejecting a generated action is fine as long as it
            # rejects identically on replay — fold the error into the stream.
            step_hashes.append(_canonical_hash({"__step_error__": repr(exc)}))
            taken.append(action)
            continue
        total_reward += reward
        step_hashes.append(_canonical_hash({
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }))
        taken.append(action)
        if terminated or truncated:
            break
    return step_hashes, taken, total_reward


def run_determinism_check(
    env,
    seed: int = 42,
    num_steps: int = 5,
    actions: list[dict] | None = None,
) -> DeterminismReport:
    """Verify the env produces identical observations across two seeded rollouts.

    Runs a rollout with `seed` recording every observation, resets, replays the
    same actions with the same seed, hashes both observation streams, and
    raises DeterminismError if the hashes differ. Leaves the env in a stepped
    state — callers must reset() before normal use.
    """
    first_hashes, taken, first_total = _rollout(env, seed, num_steps, actions)
    second_hashes, _, _second_total = _rollout(env, seed, num_steps, taken if actions is None else actions)

    first_hash = hashlib.sha256("".join(first_hashes).encode()).hexdigest()
    second_hash = hashlib.sha256("".join(second_hashes).encode()).hexdigest()

    if first_hash != second_hash:
        divergent_step = next(
            (i for i, (a, b) in enumerate(zip(first_hashes, second_hashes)) if a != b),
            None,
        )
        raise DeterminismError(seed, first_hash, second_hash, divergent_step)

    return DeterminismReport(
        passed=True,
        seed=seed,
        observation_hash=first_hash,
        actions=taken,
        total_reward=first_total,
    )
