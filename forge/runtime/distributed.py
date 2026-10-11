"""Distributed training runtime: durable queue, inference contracts, and concurrent environment scheduler (Feature Request 8)."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from forge.contracts.inference import (
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceProvider,
    InferenceRequest,
    InferenceResponse,
)
from forge.contracts.rollout import RolloutRecord
from forge.runtime.policy import seeded_random_policy

if TYPE_CHECKING:
    from forge.runtime.parallel_rollout import Policy, RolloutSpec
else:
    Policy = Any
    RolloutSpec = Any

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_TTL_S: float = 600.0  # 10 minutes TTL
DEFAULT_MAX_RETRIES: int = 3       # DLQ after 3 failures

__all__ = [
    "DurableQueueItem",
    "DurableTaskQueue",
    "EnvironmentSlot",
    "DistributedEnvironmentScheduler",
    "MockInferenceProvider",
    "DeterministicSeedPartitioner",
    "DistributedRolloutBatch",
    "DistributedParallelRolloutRunner",
    "DEFAULT_QUEUE_TTL_S",
    "DEFAULT_MAX_RETRIES",
]


@dataclass
class DurableQueueItem:
    """A durable task item with retry budget, 10-minute TTL, and DLQ constraints."""

    task_id: str
    payload: dict[str, Any]
    status: str = "queued"  # queued | running | completed | failed | truncated | dlq
    retries: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES
    wallclock_budget_s: float = 300.0
    ttl_s: float = DEFAULT_QUEUE_TTL_S
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: Any | None = None
    failure_history: list[dict[str, Any]] = field(default_factory=list)

    def is_expired(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        # 10 minute TTL since enqueue
        if (current - self.created_at) > self.ttl_s:
            return True
        # Wallclock budget since started running
        if self.started_at is not None and (current - self.started_at) > self.wallclock_budget_s:
            return True
        return False


class DurableTaskQueue:
    """Thread-safe durable task queue with retries, 10-minute TTL, and dead-letter queue (DLQ)."""

    def __init__(
        self,
        *,
        default_ttl_s: float = DEFAULT_QUEUE_TTL_S,
        default_max_retries: int = DEFAULT_MAX_RETRIES,
        max_queue_depth: int = 10000,
    ) -> None:
        self._items: dict[str, DurableQueueItem] = {}
        self._dlq: dict[str, DurableQueueItem] = {}
        self.default_ttl_s = default_ttl_s
        self.default_max_retries = default_max_retries
        self.max_queue_depth = max_queue_depth
        self._lock = threading.Lock()

    def enqueue(
        self,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
        max_retries: int | None = None,
        wallclock_budget_s: float = 300.0,
        ttl_s: float | None = None,
    ) -> DurableQueueItem:
        with self._lock:
            active_count = sum(1 for item in self._items.values() if item.status in ("queued", "running"))
            if active_count >= self.max_queue_depth:
                raise RuntimeError(
                    f"Queue backpressure limit reached ({active_count}/{self.max_queue_depth}); cannot enqueue"
                )
            tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
            item = DurableQueueItem(
                task_id=tid,
                payload=payload or {},
                max_retries=max_retries if max_retries is not None else self.default_max_retries,
                wallclock_budget_s=wallclock_budget_s,
                ttl_s=ttl_s if ttl_s is not None else self.default_ttl_s,
            )
            self._items[tid] = item
            return item

    def enqueue_batch(
        self,
        items_payload: list[dict[str, Any]],
        *,
        ttl_s: float | None = None,
        max_retries: int | None = None,
        wallclock_budget_s: float = 300.0,
    ) -> list[DurableQueueItem]:
        """Atomically enqueue a burst of tasks for bursty traffic absorption."""
        with self._lock:
            active_count = sum(1 for item in self._items.values() if item.status in ("queued", "running"))
            if active_count + len(items_payload) > self.max_queue_depth:
                raise RuntimeError(
                    f"Queue backpressure: burst of {len(items_payload)} exceeds capacity ({active_count}/{self.max_queue_depth})"
                )
            enqueued: list[DurableQueueItem] = []
            for entry in items_payload:
                tid = entry.get("task_id") or f"task_{uuid.uuid4().hex[:12]}"
                item = DurableQueueItem(
                    task_id=tid,
                    payload=entry,
                    max_retries=max_retries if max_retries is not None else self.default_max_retries,
                    wallclock_budget_s=wallclock_budget_s,
                    ttl_s=ttl_s if ttl_s is not None else self.default_ttl_s,
                )
                self._items[tid] = item
                enqueued.append(item)
            return enqueued

    def dequeue(self, worker_id: str | None = None) -> DurableQueueItem | None:
        with self._lock:
            now = time.time()
            # Sweep running tasks for wallclock budget expiration
            for item in self._items.values():
                if item.status == "running" and item.is_expired(now):
                    item.status = "truncated"
                    item.error = f"wallclock budget of {item.wallclock_budget_s}s exceeded"
                    item.completed_at = now
                elif item.status == "queued" and (now - item.created_at) > item.ttl_s:
                    item.status = "truncated"
                    item.error = f"queue TTL of {item.ttl_s}s exceeded before execution"
                    item.completed_at = now

            # Pick next queued task (matching worker_id if specified)
            for item in self._items.values():
                if item.status == "queued":
                    target_worker = item.payload.get("worker_id")
                    if worker_id is not None and target_worker is not None and target_worker != worker_id:
                        continue
                    item.status = "running"
                    item.started_at = now
                    return item
            return None

    def complete(self, task_id: str, result: Any = None) -> bool:
        with self._lock:
            item = self._items.get(task_id)
            if item is None:
                return False
            item.status = "completed"
            item.result = result
            item.completed_at = time.time()
            return True

    def fail(self, task_id: str, error: str) -> bool:
        with self._lock:
            item = self._items.get(task_id)
            if item is None:
                return False
            now = time.time()
            item.failure_history.append({
                "attempt": item.retries + 1,
                "error": error,
                "timestamp": now,
            })
            if item.is_expired(now):
                item.status = "truncated"
                item.error = f"wallclock budget of {item.wallclock_budget_s}s exceeded: {error}"
                item.completed_at = now
                return True

            if item.retries < item.max_retries:
                item.retries += 1
                item.status = "queued"
                item.started_at = None
                item.error = f"retry {item.retries}/{item.max_retries}: {error}"
                logger.info(
                    "[durable-queue] task %s failed, retrying (%d/%d): %s",
                    task_id, item.retries, item.max_retries, error,
                )
            else:
                # After max_retries failures, move to Dead Letter Queue (DLQ)
                item.status = "dlq"
                item.error = f"exceeded {item.max_retries} retries; moved to DLQ: {error}"
                item.completed_at = now
                self._dlq[task_id] = item
                logger.warning(
                    "[durable-queue] task %s moved to DLQ after %d failures: %s",
                    task_id, item.retries, error,
                )
            return True

    def get(self, task_id: str) -> DurableQueueItem | None:
        with self._lock:
            return self._items.get(task_id)

    def get_dlq(self) -> list[DurableQueueItem]:
        with self._lock:
            return list(self._dlq.values())

    def redrive_dlq(self, task_id: str | None = None) -> list[DurableQueueItem]:
        """Redrive tasks from DLQ back into the active queue with reset retries."""
        with self._lock:
            redriven: list[DurableQueueItem] = []
            targets = [task_id] if task_id else list(self._dlq.keys())
            for tid in targets:
                item = self._dlq.pop(tid, None)
                if item is not None:
                    item.status = "queued"
                    item.retries = 0
                    item.started_at = None
                    item.error = None
                    item.completed_at = None
                    self._items[tid] = item
                    redriven.append(item)
            return redriven

    def list_all(self) -> list[DurableQueueItem]:
        with self._lock:
            return list(self._items.values())

    def size(self) -> int:
        with self._lock:
            return sum(1 for item in self._items.values() if item.status in ("queued", "running"))

    def stats(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {
                "queued": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "truncated": 0,
                "dlq": len(self._dlq),
                "total": len(self._items),
            }
            for item in self._items.values():
                counts[item.status] = counts.get(item.status, 0) + 1
            return counts


# ---------------------------------------------------------------------------
# Distributed Parallel Rollouts & Deterministic Partitioning
# ---------------------------------------------------------------------------

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
        from forge.runtime.parallel_rollout import RolloutSpec
        specs = [
            RolloutSpec(seed=seed, task=task, policy=policy)
            for seed in range(seed_start, seed_start + num_rollouts)
        ]
        return self.run(specs, run_id=run_id, timeout_s=timeout_s)

    def _run_one(self, spec: RolloutSpec) -> RolloutRecord:
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

        env = self.env_factory()
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
            raise exc
        finally:
            if hasattr(env, "close"):
                env.close()

        return RolloutRecord(
            seed=spec.seed,
            episode_id=episode_id or f"rollout-{spec.seed}",
            outcome=self._classify(passed, total_reward, invalid_actions, error),
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

    @staticmethod
    def _classify(
        passed: bool, total_reward: float, invalid_actions: int, error: str | None
    ) -> str:
        if error is not None:
            return "edge_case"
        if passed:
            return "success"
        if invalid_actions > 0:
            return "edge_case"
        if total_reward > 0:
            return "partial_success"
        return "failure"



@dataclass
class EnvironmentSlot:
    """An allocated slot for an environment instance."""

    slot_id: str
    env_name: str
    status: str = "idle"  # idle | building | allocated | error
    container_id: str | None = None
    port: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DistributedEnvironmentScheduler:
    """Scheduler that manages concurrent building and allocation of multiple environments."""

    def __init__(self, max_concurrent_environments: int = 4) -> None:
        self.max_concurrent_environments = max(1, max_concurrent_environments)
        self._slots: dict[str, EnvironmentSlot] = {}
        self._lock = threading.Lock()

    def build_environment(
        self,
        env_name: str,
        build_fn: Callable[[str], tuple[str | None, int | None]] | None = None,
    ) -> EnvironmentSlot:
        with self._lock:
            active_count = sum(
                1 for s in self._slots.values() if s.status in ("building", "allocated")
            )
            if active_count >= self.max_concurrent_environments:
                raise RuntimeError(
                    f"Concurrency limit reached: {active_count}/{self.max_concurrent_environments} environments active"
                )

            slot_id = f"slot_{env_name}_{uuid.uuid4().hex[:6]}"
            slot = EnvironmentSlot(slot_id=slot_id, env_name=env_name, status="building")
            self._slots[slot_id] = slot

        try:
            cid, port = (None, None)
            if build_fn is not None:
                cid, port = build_fn(env_name)
            with self._lock:
                slot.container_id = cid
                slot.port = port
                slot.status = "idle"
            return slot
        except Exception as exc:
            with self._lock:
                slot.status = "error"
                slot.metadata["error"] = str(exc)
            raise

    def schedule_concurrent_builds(
        self,
        env_names: list[str],
        build_fn: Callable[[str], tuple[str | None, int | None]] | None = None,
    ) -> list[EnvironmentSlot]:
        """Build multiple environments concurrently up to the scheduler limit."""
        results: list[EnvironmentSlot] = []
        threads: list[threading.Thread] = []
        errors: list[Exception] = []

        def _worker(name: str):
            try:
                slot = self.build_environment(name, build_fn=build_fn)
                results.append(slot)
            except Exception as e:
                errors.append(e)

        for name in env_names:
            t = threading.Thread(target=_worker, args=(name,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

    def acquire(self, env_name: str) -> EnvironmentSlot | None:
        with self._lock:
            for slot in self._slots.values():
                if slot.env_name == env_name and slot.status == "idle":
                    slot.status = "allocated"
                    return slot
            return None

    def release(self, slot_id: str) -> bool:
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None:
                return False
            slot.status = "idle"
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            total_slots = len(self._slots)
            allocated = sum(1 for s in self._slots.values() if s.status == "allocated")
            idle = sum(1 for s in self._slots.values() if s.status == "idle")
            building = sum(1 for s in self._slots.values() if s.status == "building")
            return {
                "max_concurrent_environments": self.max_concurrent_environments,
                "total_slots": total_slots,
                "allocated": allocated,
                "idle": idle,
                "building": building,
                "slots": [
                    {
                        "slot_id": s.slot_id,
                        "env_name": s.env_name,
                        "status": s.status,
                        "container_id": s.container_id,
                        "port": s.port,
                    }
                    for s in self._slots.values()
                ],
            }


class MockInferenceProvider(InferenceProvider):
    """Inference provider implementation executing local or mock inference requests."""

    def __init__(self, default_response: str = "Action completed.") -> None:
        self.default_response = default_response

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        t0 = time.time()
        content = self.default_response
        if request.prompt:
            content = f"Processed prompt: {request.prompt[:60]}"
        elif request.messages:
            last = request.messages[-1].get("content", "")
            content = f"Response to: {last[:60]}"
        latency = (time.time() - t0) * 1000.0
        return InferenceResponse(
            content=content,
            model=request.model,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            latency_ms=latency,
        )

    def generate_batch(self, batch: InferenceBatchRequest) -> InferenceBatchResponse:
        return InferenceBatchResponse(
            responses=[self.generate(req) for req in batch.requests]
        )
