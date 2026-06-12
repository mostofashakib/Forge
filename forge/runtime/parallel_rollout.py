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

from forge.runtime.policy import seeded_random_policy

Policy = Callable[[dict, frozenset], dict]


@dataclass
class RolloutSpec:
    seed: int
    task: dict | None = None
    policy: Policy | None = None  # None → seeded-random over the env's tools


@dataclass
class RolloutRecord:
    seed: int
    episode_id: str | None
    outcome: str  # "success" | "failure" | "partial_success" | "edge_case"
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    invalid_actions: int
    error: str | None = None


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
        episode_id = None
        total_reward = 0.0
        steps = 0
        invalid_actions = 0
        terminated = truncated = False
        error: str | None = None

        env = self._env_factory()
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
                    break
        except Exception as exc:
            error = repr(exc)
        finally:
            env.close()

        return RolloutRecord(
            seed=spec.seed,
            episode_id=episode_id,
            outcome=self._classify(terminated, total_reward, invalid_actions, error),
            total_reward=total_reward,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            invalid_actions=invalid_actions,
            error=error,
        )

    @staticmethod
    def _classify(
        terminated: bool, total_reward: float, invalid_actions: int, error: str | None
    ) -> str:
        if error is not None:
            return "edge_case"
        if terminated:
            return "success"
        if invalid_actions > 0:
            return "edge_case"
        if total_reward > 0:
            return "partial_success"
        return "failure"
