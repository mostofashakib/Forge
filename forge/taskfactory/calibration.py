"""How often a target model solves each task, and what that says about the batch.

A task the model solved on every attempt is too easy, one it never solved
is unsolved, and the rest are the frontier. A task it never solved, where
every attempt finished and missed the same checks, is suspect: those checks
may test something the objective never asks for.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Band = Literal["too_easy", "frontier", "unsolved", "suspect"]
BANDS: tuple[Band, ...] = ("too_easy", "frontier", "unsolved", "suspect")
# At most five attempts per task.
MIN_ATTEMPTS, MAX_ATTEMPTS = 1, 5


@dataclass(frozen=True)
class TaskRate:
    task_id: str
    category: str
    difficulty: int
    passed: int
    attempts: int
    suspect: bool = False

    def __post_init__(self) -> None:
        if self.suspect and self.passed:
            raise ValueError(f"{self.task_id} passed {self.passed} attempts, so its checks are not suspect")

    @property
    def solve_rate(self) -> float:
        return self.passed / self.attempts

    @property
    def band(self) -> Band:
        return "suspect" if self.suspect else band(self.passed, self.attempts)


def attempt_count_problem(attempts: int) -> str | None:
    if MIN_ATTEMPTS <= attempts <= MAX_ATTEMPTS:
        return None
    return f"attempts per task must be {MIN_ATTEMPTS} to {MAX_ATTEMPTS}, got {attempts}"


def band(passed: int, attempts: int) -> Band:
    if attempts < 1 or not 0 <= passed <= attempts:
        raise ValueError(f"{passed} passes out of {attempts} attempts is not a result")
    if passed == attempts:
        return "too_easy"
    return "unsolved" if passed == 0 else "frontier"


def calibration_summary(rates: Sequence[TaskRate]) -> dict:
    """Band counts overall, by category, and by difficulty, plus the solve-rate histogram."""
    attempt_counts = {rate.attempts for rate in rates}
    if len(attempt_counts) > 1:
        raise ValueError(f"tasks have different attempt counts {sorted(attempt_counts)}, so rates do not compare")
    histogram: dict[str, int] = {}
    if attempt_counts:
        attempts = attempt_counts.pop()
        histogram = {_rate_key(passed / attempts): 0 for passed in range(attempts + 1)}
        for rate in rates:
            histogram[_rate_key(rate.solve_rate)] += 1
    return {
        "tasks": len(rates),
        "bands": _band_counts(rates),
        "by_category": _grouped(rates, lambda rate: rate.category),
        "by_difficulty": _grouped(rates, lambda rate: str(rate.difficulty)),
        "histogram": histogram,
    }


def _band_counts(rates: Sequence[TaskRate]) -> dict[str, int]:
    counts = dict.fromkeys(BANDS, 0)
    for rate in rates:
        counts[rate.band] += 1
    return counts


def _grouped(rates: Sequence[TaskRate], key) -> dict[str, dict[str, int]]:
    groups: dict[str, list[TaskRate]] = {}
    for rate in rates:
        groups.setdefault(key(rate), []).append(rate)
    return {name: _band_counts(members) for name, members in sorted(groups.items())}


def _rate_key(rate: float) -> str:
    return f"{rate:.1f}"
