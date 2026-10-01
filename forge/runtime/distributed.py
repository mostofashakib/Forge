"""Distributed training runtime: durable queue, inference contracts, and concurrent environment scheduler (Feature Request 8)."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from forge.contracts.inference import (
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceProvider,
    InferenceRequest,
    InferenceResponse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DurableQueueItem",
    "DurableTaskQueue",
    "EnvironmentSlot",
    "DistributedEnvironmentScheduler",
    "MockInferenceProvider",
]


@dataclass
class DurableQueueItem:
    """A durable task item with retry budget and wallclock timeout constraints."""

    task_id: str
    payload: dict[str, Any]
    status: str = "queued"  # queued | running | completed | failed | truncated
    retries: int = 0
    max_retries: int = 3
    wallclock_budget_s: float = 300.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: Any | None = None

    def is_expired(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        if self.started_at is None:
            return False
        return (current - self.started_at) > self.wallclock_budget_s


class DurableTaskQueue:
    """Thread-safe durable task queue with retries and wallclock budgeting."""

    def __init__(self) -> None:
        self._items: dict[str, DurableQueueItem] = {}
        self._lock = threading.Lock()

    def enqueue(
        self,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
        max_retries: int = 3,
        wallclock_budget_s: float = 300.0,
    ) -> DurableQueueItem:
        with self._lock:
            tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
            item = DurableQueueItem(
                task_id=tid,
                payload=payload or {},
                max_retries=max_retries,
                wallclock_budget_s=wallclock_budget_s,
            )
            self._items[tid] = item
            return item

    def dequeue(self) -> DurableQueueItem | None:
        with self._lock:
            now = time.time()
            # First sweep running tasks for wallclock budget expiration
            for item in self._items.values():
                if item.status == "running" and item.is_expired(now):
                    item.status = "truncated"
                    item.error = f"wallclock budget of {item.wallclock_budget_s}s exceeded"
                    item.completed_at = now

            # Pick next queued task
            for item in self._items.values():
                if item.status == "queued":
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
                item.status = "failed"
                item.error = error
                item.completed_at = now
            return True

    def get(self, task_id: str) -> DurableQueueItem | None:
        with self._lock:
            return self._items.get(task_id)

    def list_all(self) -> list[DurableQueueItem]:
        with self._lock:
            return list(self._items.values())

    def size(self) -> int:
        with self._lock:
            return sum(1 for item in self._items.values() if item.status in ("queued", "running"))


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
