from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

from forge.runtime.canonical import canonical_hash
from forge.runtime.errors import DeterminismError
from forge.runtime.policy import seeded_random_policy

__all__ = [
    "DeterminismError",
    "DeterminismReport",
    "run_determinism_check",
    "check_fingerprint_determinism",
]


@dataclass
class DeterminismReport:
    passed: bool
    seed: int
    observation_hash: str
    actions: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    fingerprint: str = ""
    recreated_fingerprint: str = ""
    fingerprint_matched: bool = True
    task: str | None = None


def check_fingerprint_determinism(
    env,
    task: Any = None,
    seed: int = 42,
    recreate_fn: Callable[..., Any] | None = None,
) -> tuple[str, str, bool]:
    """Compare fingerprint generated during initial environment creation with recreated snapshot fingerprint.

    For each (task, seed) pair, these two fingerprints must match exactly.
    """
    options = {"task": task} if task is not None else None
    if hasattr(env, "reset"):
        try:
            env.reset(seed=seed, options=options)
        except TypeError:
            env.reset(seed=seed)

    initial_fp = env.fingerprint() if hasattr(env, "fingerprint") else ""

    if recreate_fn is not None:
        try:
            recreated_env = recreate_fn(task=task, seed=seed)
        except TypeError:
            recreated_env = recreate_fn()
        recreated_fp = (
            recreated_env.fingerprint() if hasattr(recreated_env, "fingerprint") else ""
        )
    elif hasattr(env, "snapshot") and hasattr(env, "restore_snapshot"):
        snap = env.snapshot()
        env.restore_snapshot(snap)
        recreated_fp = env.fingerprint() if hasattr(env, "fingerprint") else ""
    elif hasattr(env, "reset"):
        try:
            env.reset(seed=seed, options=options)
        except TypeError:
            env.reset(seed=seed)
        recreated_fp = env.fingerprint() if hasattr(env, "fingerprint") else ""
    else:
        recreated_fp = initial_fp

    if initial_fp != recreated_fp:
        raise DeterminismError(
            seed,
            initial_fp,
            recreated_fp,
            reason=f"fingerprint mismatch for (task={task}, seed={seed}): {initial_fp} != {recreated_fp}",
        )

    return initial_fp, recreated_fp, True


def _rollout(
    env, seed: int, num_steps: int, actions: list[dict] | None
) -> tuple[list[str], list[dict], float]:
    """Reset with `seed`, run up to `num_steps`, return per-step hashes, actions
    taken, and the total reward.

    Each step's hash covers the observation, reward, and termination flags, so
    a divergent score fails the check just like a divergent observation. If
    `actions` is None, actions come from the shared seeded-random policy, so
    both rollouts can regenerate or replay the identical sequence.
    """
    obs, _info = env.reset(seed=seed)
    step_hashes = [canonical_hash(obs)]
    taken: list[dict] = []
    total_reward = 0.0
    ended = False

    if actions is None:
        if not env.action_types:
            return step_hashes, taken, total_reward
        policy = seeded_random_policy(seed)
        planned = [policy(obs, env.action_types) for _ in range(num_steps)]
    else:
        planned = actions[:num_steps]

    for action in planned:
        try:
            obs, reward, terminated, truncated, _info = env.step(action)
        except Exception as exc:
            # A transition rejecting a generated action is fine as long as it
            # rejects identically on replay — fold the error into the stream.
            step_hashes.append(canonical_hash({"__step_error__": repr(exc)}))
            taken.append(action)
            continue
        total_reward += reward
        step_hashes.append(canonical_hash({
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }))
        taken.append(action)
        if terminated or truncated:
            ended = True
            break
    if not ended and hasattr(env, "finalize_episode"):
        try:
            evaluation = env.finalize_episode("determinism_probe")
        except NotImplementedError:
            evaluation = None
        if evaluation is not None:
            total_reward = evaluation.total_reward
            step_hashes.append(canonical_hash({
                "final_reward": evaluation.total_reward,
                "passed": evaluation.passed,
                "verification_results": [
                    result.model_dump()
                    for result in evaluation.verification_results
                ],
            }))
    return step_hashes, taken, total_reward


def run_determinism_check(
    env,
    seed: int = 42,
    num_steps: int = 5,
    actions: list[dict] | None = None,
    task: Any = None,
    recreate_fn: Callable[..., Any] | None = None,
) -> DeterminismReport:
    """Verify the env produces identical observations across two seeded rollouts
    and identical fingerprints between creation and snapshot recreation.

    Runs a rollout with `seed` recording every observation, resets, replays the
    same actions with the same seed, hashes both observation streams, checks
    environment fingerprint consistency for `(task, seed)` pairs, and raises
    DeterminismError if hashes or fingerprints differ.
    """
    initial_fp, recreated_fp, fp_ok = check_fingerprint_determinism(
        env, task=task, seed=seed, recreate_fn=recreate_fn
    )

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
        fingerprint=initial_fp,
        recreated_fingerprint=recreated_fp,
        fingerprint_matched=fp_ok,
        task=str(task) if task is not None else None,
    )
