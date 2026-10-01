"""The slot plan turns a taxonomy into exactly the number of tasks asked for.

Code, not the LLM, decides the plan, so the count is exact and the spread
across difficulties and categories is even and reproducible.
"""
from __future__ import annotations

from collections import Counter

import pytest

from forge.taskfactory.schemas import Taxonomy, TaxonomyCategory
from forge.taskfactory.slots import plan_slots


def _taxonomy(*categories: tuple[str, list[int]]) -> Taxonomy:
    return Taxonomy(
        categories=[
            TaxonomyCategory(name=name, description=f"{name} work", exercises=[], difficulties=levels)
            for name, levels in categories
        ],
        difficulty_rubric={"1": "one step", "3": "a few dependent steps", "5": "long horizon"},
    )


TAXONOMY = _taxonomy(
    ("triage", [1, 2, 3]),
    ("reconcile", [3, 4, 5]),
    ("cleanup", [1, 2, 4, 5]),
    ("report", [2, 3, 5]),
)


@pytest.mark.parametrize("count", [1, 7, 30, 100])
def test_the_plan_has_exactly_the_requested_count(count):
    assert len(plan_slots(TAXONOMY, count)) == count


def test_every_slot_pairs_a_category_with_a_level_it_covers():
    covered = {c.name: set(c.difficulties) for c in TAXONOMY.categories}

    for slot in plan_slots(TAXONOMY, 40):
        assert slot.difficulty in covered[slot.category]


def test_difficulty_levels_are_spread_evenly():
    levels = Counter(slot.difficulty for slot in plan_slots(TAXONOMY, 50))

    assert set(levels) == {1, 2, 3, 4, 5}
    assert max(levels.values()) - min(levels.values()) <= 1


def test_categories_rotate_within_a_level():
    level_five = [s.category for s in plan_slots(TAXONOMY, 50) if s.difficulty == 5]

    # reconcile, cleanup and report all cover level 5, so no one of them
    # takes every level-5 slot.
    assert set(level_five) == {"reconcile", "cleanup", "report"}


def test_the_plan_is_reproducible_and_indexed():
    first = plan_slots(TAXONOMY, 23)

    assert first == plan_slots(TAXONOMY, 23)
    assert [s.index for s in first] == list(range(23))


def test_a_non_positive_count_is_rejected():
    with pytest.raises(ValueError):
        plan_slots(TAXONOMY, 0)
