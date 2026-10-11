"""Distributed parallel rollout runner with worker pools, queues, and run aggregation."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from forge.contracts.rollout import RolloutRecord
from forge.runtime.distributed.partitioner import DeterministicSeedPartitioner
from forge.runtime.distributed.queue import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_QUEUE_TTL_S,
    DurableQueueItem,
    DurableTaskQueue,
)
from forge.runtime.parallel_rollout import (
    Policy,
    RolloutSpec,
    classify_rollout_outcome,
    execute_single_rollout,
)


@dataclass
class DistributedRolloutBatch:
    """Run-level aggregation across distributed worker rollout executions."""

    run_id: str
    records: list[RolloutRecord] = field(default_factory=list)
    dlq_items: list[DurableQueueItem] = field(default_factory=list)
    total_scheduled: int = 0
    duration_s: float = 0.0
    worker_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.outcome] = counts.get(record.outcome, 0) + 1
        return counts

    @property
    def pass_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.passed) / len(self.records)

    @property
    def mean_reward(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.total_reward for r in self.records) / len(self.records)

    def by_outcome(self, outcome: str) -> list[RolloutRecord]:
        return [r for r in self.records if r.outcome == outcome]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_scheduled": self.total_scheduled,
            "completed": len(self.records),
            "dlq_count": len(self.dlq_items),
            "outcome_counts": self.outcome_counts,
            "pass_rate": round(self.pass_rate, 4),
            "mean_reward": round(self.mean_reward, 4),
            "duration_s": round(self.duration_s, 2),
            "worker_distribution": dict(self.worker_distribution),
        }


class DistributedParallelRolloutRunner:
    """Runs parallel rollouts across multiple workers with burst queue, retries, DLQ, and 10m TTL."""

    def __init__(
        self,
        env_factory: Callable[[], object],
        *,
        num_workers: int = 4,
        queue: DurableTaskQueue | None = None,
        ttl_s: float = DEFAULT_QUEUE_TTL_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_queue_depth: int = 10000,
    ) -> None:
        self.env_factory = env_factory
        self.num_workers = max(1, num_workers)
        self.queue = queue or DurableTaskQueue(
            default_ttl_s=ttl_s,
            default_max_retries=max_retries,
            max_queue_depth=max_queue_depth,
        )
        self.ttl_s = ttl_s
        self.max_retries = max_retries

    def run(
        self,
        specs: list[RolloutSpec],
        *,
        run_id: str | None = None,
        timeout_s: float | None = None,
    ) -> DistributedRolloutBatch:
        rid = run_id or f"run_{uuid.uuid4().hex[:10]}"
        t0 = time.time()

        tasks_payload = []
        for spec in specs:
            worker_id = DeterministicSeedPartitioner.assign_worker(spec.seed, self.num_workers)
            tasks_payload.append({
                "task_id": f"{rid}_s{spec.seed}",
                "run_id": rid,
                "worker_id": f"worker_{worker_id}",
                "seed": spec.seed,
                "spec": spec,
            })

        self.queue.enqueue_batch(tasks_payload, ttl_s=self.ttl_s, max_retries=self.max_retries)

        results: list[RolloutRecord] = []
        worker_distribution: dict[str, int] = {}
        results_lock = threading.Lock()
        stop_event = threading.Event()

        def _worker(worker_idx: int):
            wid = f"worker_{worker_idx}"
            while not stop_event.is_set():
                item = self.queue.dequeue(worker_id=wid)
                if item is None:
                    with self.queue._lock:
                        run_active = any(
                            it.payload.get("run_id") == rid and it.status in ("queued", "running")
                            for it in self.queue._items.values()
                        )
                    if not run_active:
                        break
                    time.sleep(0.005)
                    continue

                if item.payload.get("run_id") != rid:
                    continue

                spec = item.payload["spec"]
                try:
                    record = self._run_one(spec)
                    self.queue.complete(item.task_id, result=record)
                    with results_lock:
                        results.append(record)
                        worker_distribution[wid] = worker_distribution.get(wid, 0) + 1
                except Exception as exc:
                    self.queue.fail(item.task_id, error=str(exc))

        threads = [threading.Thread(target=_worker, args=(w,)) for w in range(self.num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout_s)
        stop_event.set()

        dlq_items = [item for item in self.queue.get_dlq() if item.payload.get("run_id") == rid]
        duration = time.time() - t0

        return DistributedRolloutBatch(
            run_id=rid,
            records=results,
            dlq_items=dlq_items,
            total_scheduled=len(specs),
            duration_s=duration,
            worker_distribution=worker_distribution,
        )

    def run_distributed(
        self,
        task: dict | None,
        num_rollouts: int,
        *,
        seed_start: int = 0,
        policy: Policy | None = None,
        run_id: str | None = None,
        timeout_s: float | None = None,
    ) -> DistributedRolloutBatch:
        specs = [
            RolloutSpec(seed=seed, task=task, policy=policy)
            for seed in range(seed_start, seed_start + num_rollouts)
        ]
        return self.run(specs, run_id=run_id, timeout_s=timeout_s)

    def _run_one(self, spec: RolloutSpec) -> RolloutRecord:
        return execute_single_rollout(self.env_factory, spec, propagate_exception=True)

    _classify = staticmethod(classify_rollout_outcome)
