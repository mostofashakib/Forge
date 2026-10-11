"""Parallel rollouts: many isolated copies of the same task at once.

Each rollout gets its own environment instance from a factory — never a
shared one — so in-process envs start in milliseconds, run concurrently
without interference, and tear down by going out of scope. Outcomes are
classified (success / failure / partial_success / edge_case) so one batch
over one task yields the diverse scenarios RL training needs.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from forge.contracts import RolloutRecord
from forge.runtime.policy import seeded_random_policy

Policy = Callable[[dict, frozenset], dict]


@dataclass
class RolloutSpec:
    seed: int
    task: dict | None = None
    policy: Policy | None = None  # None → seeded-random over the env's tools


@dataclass
class RolloutBatch:
    records: list[RolloutRecord] = field(default_factory=list)

    @property
    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.outcome] = counts.get(record.outcome, 0) + 1
        return counts

    def by_outcome(self, outcome: str) -> list[RolloutRecord]:
        return [r for r in self.records if r.outcome == outcome]


def classify_rollout_outcome(
    passed: bool, total_reward: float, invalid_actions: int, error: str | None
) -> str:
    """Classify the rollout outcome into a standardized taxonomy bucket."""
    if error is not None:
        return "edge_case"
    if passed:
        return "success"
    if invalid_actions > 0:
        return "edge_case"
    if total_reward > 0:
        return "partial_success"
    return "failure"


def execute_single_rollout(
    env_factory: Callable[[], object],
    spec: RolloutSpec,
    *,
    propagate_exception: bool = False,
) -> RolloutRecord:
    """Execute a single episode rollout using an isolated environment instance."""
    episode_id = None
    total_reward = 0.0
    steps = 0
    invalid_actions = 0
    terminated = truncated = False
    passed = False
    termination_reason = "unknown"
    verification_results: list[dict] = []
    reward_breakdown: dict = {}
    error: str | None = None

    env = env_factory()
    policy = spec.policy or seeded_random_policy(spec.seed)
    try:
        options = {"task": spec.task} if spec.task is not None else None
        obs, info = env.reset(seed=spec.seed, options=options)
        episode_id = info.get("episode_id")
        for _ in range(env.env_spec.max_steps):
            action = policy(obs, env.action_types)
            obs, reward, terminated, truncated, step_info = env.step(action)
            steps += 1
            total_reward += reward
            if "error" in step_info:
                invalid_actions += 1
            if terminated or truncated:
                passed = bool(step_info.get("passed", False))
                termination_reason = step_info.get("termination_reason") or "unknown"
                verification_results = step_info.get("verifier_results", [])
                reward_breakdown = step_info.get("reward_breakdown", {})
                break
    except Exception as exc:
        error = repr(exc)
        if propagate_exception:
            raise exc
    finally:
        if hasattr(env, "close"):
            env.close()

    return RolloutRecord(
        seed=spec.seed,
        episode_id=episode_id or f"rollout-{spec.seed}",
        outcome=classify_rollout_outcome(passed, total_reward, invalid_actions, error),
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        invalid_actions=invalid_actions,
        error=error,
        passed=passed,
        termination_reason=termination_reason,
        verification_results=verification_results,
        reward_breakdown=reward_breakdown,
    )


class ParallelRolloutRunner:
    """Runs many rollouts of the same task on isolated env copies.

    `env_factory` must return a *fresh* environment every call; the runner
    builds one per rollout inside the worker, runs the episode, and closes
    it. A crashing rollout becomes an `edge_case` record instead of sinking
    the batch.
    """

    def __init__(self, env_factory: Callable[[], object], max_workers: int = 8) -> None:
        self._env_factory = env_factory
        self._max_workers = max_workers

    def run(self, specs: list[RolloutSpec]) -> RolloutBatch:
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(self._run_one, spec) for spec in specs]
            records = [future.result() for future in futures]
        return RolloutBatch(records=records)

    def run_diverse(
        self,
        task: dict | None,
        num_rollouts: int,
        seed_start: int = 0,
        policy: Policy | None = None,
    ) -> RolloutBatch:
        """Same task across a seed range — different initial worlds, one batch."""
        return self.run([
            RolloutSpec(seed=seed, task=task, policy=policy)
            for seed in range(seed_start, seed_start + num_rollouts)
        ])

    def _run_one(self, spec: RolloutSpec) -> RolloutRecord:
        return execute_single_rollout(self._env_factory, spec, propagate_exception=False)

    _classify = staticmethod(classify_rollout_outcome)


def create_rollout_runner(
    env_factory: Callable[[], object],
    mode: str = "local",
    *,
    max_workers: int = 8,
    num_workers: int = 4,
    queue: Any = None,
    ttl_s: float = 600.0,
    max_retries: int = 3,
    max_queue_depth: int = 10000,
) -> ParallelRolloutRunner | Any:
    """Create a rollout runner supporting both local and distributed execution modes.

    Args:
        env_factory: Factory returning a fresh environment instance per rollout.
        mode: "local" for thread pool on a single host;
              "distributed" for multi-worker durable queue execution.
        max_workers: Worker thread count for local mode.
        num_workers: Worker pool size for distributed mode.
        queue: Optional DurableTaskQueue instance for distributed mode.
        ttl_s: Task TTL in seconds (default 10 minutes = 600s).
        max_retries: Retry attempts before DLQ (default 3).
        max_queue_depth: Maximum queue depth for backpressure protection.

    Returns:
        ParallelRolloutRunner if mode="local",
        DistributedParallelRolloutRunner if mode="distributed".
    """
    if mode == "local":
        return ParallelRolloutRunner(env_factory=env_factory, max_workers=max_workers)
    elif mode == "distributed":
        from forge.runtime.distributed import DistributedParallelRolloutRunner

        return DistributedParallelRolloutRunner(
            env_factory=env_factory,
            num_workers=num_workers,
            queue=queue,
            ttl_s=ttl_s,
            max_retries=max_retries,
            max_queue_depth=max_queue_depth,
        )
    else:
        raise ValueError(
            f"Unsupported rollout mode: {mode!r}. Expected 'local' or 'distributed'."
        )


def run_parallel_rollouts(
    env_factory: Callable[[], object],
    task: dict | None,
    num_rollouts: int,
    *,
    seed_start: int = 0,
    policy: Policy | None = None,
    mode: str = "local",
    **kwargs: Any,
) -> RolloutBatch | Any:
    """Run parallel rollouts using either local or distributed execution mode."""
    runner = create_rollout_runner(env_factory, mode=mode, **kwargs)
    return runner.run_diverse(task, num_rollouts, seed_start=seed_start, policy=policy)


def __getattr__(name: str):
    if name in ("DistributedParallelRolloutRunner", "DistributedRolloutBatch"):
        import forge.runtime.distributed as dist

        return getattr(dist, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


