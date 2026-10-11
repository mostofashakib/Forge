"""Distributed environment scheduler for managing concurrent building and allocation."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


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
