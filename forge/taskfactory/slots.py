"""Turn a taxonomy into an exact, evenly spread plan of task slots."""
from __future__ import annotations

from dataclasses import dataclass

from forge.taskfactory.schemas import Taxonomy


@dataclass(frozen=True)
class Slot:
    index: int
    category: str
    difficulty: int


def plan_slots(taxonomy: Taxonomy, count: int) -> list[Slot]:
    """Plan `count` slots, cycling through difficulty levels, then categories.

    Each pass takes one level in turn, and within a level the categories that
    cover it take turns. Levels therefore differ by at most one slot, and no
    category monopolises a level.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
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
    return slots
