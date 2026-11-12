"""Turn a taxonomy into an exact, evenly spread plan of task slots."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from forge.taskfactory.schemas import EDGE_CASES, FAMILY_EDGE_CASES, Taxonomy


@dataclass(frozen=True)
class Slot:
    index: int
    category: str
    difficulty: int
    edge_case: str | None = None


def plan_slots(taxonomy: Taxonomy, count: int, edge_cases: Sequence[str] = ()) -> list[Slot]:
    """Plan `count` slots, cycling through difficulty levels, then categories.

    Each pass takes one level in turn, and within a level the categories that
    cover it take turns. Levels therefore differ by at most one slot, and no
    category monopolises a level. Requested edge cases then share the slots
    with plain slots (see `_with_edge_cases`).
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    problems = edge_case_problems(edge_cases)
    if problems:
        raise ValueError("; ".join(problems))
    by_level: dict[int, list[str]] = {}
    for category in taxonomy.categories:
        for level in sorted(set(category.difficulties)):
            by_level.setdefault(level, []).append(category.name)
    if not by_level:
        raise ValueError("taxonomy covers no difficulty levels")
    levels = sorted(by_level)
    turns = {level: 0 for level in levels}
    slots: list[Slot] = []
    while len(slots) < count:
        level = levels[len(slots) % len(levels)]
        names = by_level[level]
        slots.append(Slot(index=len(slots), category=names[turns[level] % len(names)], difficulty=level))
        turns[level] += 1
    return _with_edge_cases(slots, edge_cases) if edge_cases else slots


def edge_case_problems(edge_cases: Sequence[str], family: str | None = None) -> list[str]:
    """Every reason a request for these edge cases cannot be planned. Empty means it can."""
    problems = []
    unknown = [case for case in edge_cases if case not in EDGE_CASES]
    if unknown:
        problems.append(f"unknown edge cases {unknown}, use any of {list(EDGE_CASES)}")
    repeated = sorted({case for case in edge_cases if list(edge_cases).count(case) > 1})
    if repeated:
        problems.append(f"edge cases {repeated} are requested more than once")
    if family is not None:
        unoffered = [case for case in edge_cases if case in EDGE_CASES and case not in FAMILY_EDGE_CASES[family]]
        if unoffered:
            problems.append(f"edge cases {unoffered} do not apply to {family} environments")
    return problems


def _with_edge_cases(slots: list[Slot], edge_cases: Sequence[str]) -> list[Slot]:
    """Give each slot one requested edge case or none, in turn.

    Turns are counted per category and level, so every pairing sees every
    option. Each pairing starts one option further along than the one before,
    so pairings with a single slot do not all get the same option.
    """
    options: list[str | None] = [*edge_cases, None]
    start: dict[tuple[str, int], int] = {}
    taken: dict[tuple[str, int], int] = {}
    planned = []
    for slot in slots:
        key = (slot.category, slot.difficulty)
        start.setdefault(key, len(start))
        turn = start[key] + taken.get(key, 0)
        taken[key] = taken.get(key, 0) + 1
        planned.append(replace(slot, edge_case=options[turn % len(options)]))
    return planned
