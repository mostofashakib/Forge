"""Deterministic seed partitioning across distributed workers."""
from __future__ import annotations


class DeterministicSeedPartitioner:
    """Deterministically assigns seeds across workers to ensure reproducibility."""

    @staticmethod
    def partition_seeds(
        base_seed: int,
        num_rollouts: int,
        num_workers: int,
    ) -> dict[int, list[int]]:
        """Round-robin partition of seeds across worker indices."""
        partitions: dict[int, list[int]] = {w: [] for w in range(num_workers)}
        for index in range(num_rollouts):
            worker_idx = index % num_workers
            partitions[worker_idx].append(base_seed + index)
        return partitions

    @staticmethod
    def assign_worker(seed: int, num_workers: int, base_seed: int = 0) -> int:
        """Deterministic worker mapping for a given seed."""
        return (seed - base_seed) % num_workers
