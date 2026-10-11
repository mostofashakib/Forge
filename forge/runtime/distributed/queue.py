"""Durable task queue with retry budget, 10-minute TTL, and dead-letter queue (DLQ)."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_TTL_S: float = 600.0  # 10 minutes TTL
DEFAULT_MAX_RETRIES: int = 3       # DLQ after 3 failures


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
