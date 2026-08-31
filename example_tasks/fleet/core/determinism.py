"""Deterministic time, randomness, and ID generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import uuid


@dataclass
class VirtualClock:
    start_ms: int = 1_700_000_000_000
    step_ms: int = 1_000
    ticks: int = 0

    def now_ms(self) -> int:
        return self.start_ms + (self.ticks * self.step_ms)

    def tick(self) -> int:
        self.ticks += 1
        return self.now_ms()

    def reset(self) -> None:
        self.ticks = 0


class DeterministicIdGenerator:
    def __init__(self, seed: int, namespace: str) -> None:
        self._seed = seed
        self._namespace = namespace
        self._counters: dict[str, int] = {}
        self._user_counter = 0

    def next(self, prefix: str) -> str:
        counter = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = counter
        raw = f"{self._seed}:{self._namespace}:{prefix}:{counter}".encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:24]
        return f"{prefix}_{digest}"

    def uuid(self, label: str) -> str:
        if label == "user":
            self._user_counter += 1
            return str(uuid.UUID(int=self._user_counter))
        return str(uuid.UUID(hashlib.md5(self.next(label).encode("utf-8")).hexdigest()))

    def reset(self) -> None:
        self._counters.clear()
        self._user_counter = 0


class SeededRandom:
    def __init__(self, seed: int, namespace: str) -> None:
        self._seed = seed
        self._namespace = namespace
        self._counter = 0

    def _next_val(self) -> int:
        self._counter += 1
        raw = f"{self._seed}:{self._namespace}:{self._counter}".encode("utf-8")
        return int(hashlib.sha256(raw).hexdigest(), 16)

    def choice(self, values: list[str]) -> str:
        if not values:
            raise ValueError("Cannot choose from empty list")
        idx = self._next_val() % len(values)
        return values[idx]

    def randint(self, lower: int, upper: int) -> int:
        if lower > upper:
            raise ValueError("lower cannot be greater than upper")
        span = upper - lower + 1
        return lower + (self._next_val() % span)


