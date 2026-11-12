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


def test_without_edge_cases_no_slot_gets_one():
    assert all(slot.edge_case is None for slot in plan_slots(TAXONOMY, 30))


def test_edge_cases_leave_the_rest_of_the_plan_unchanged():
    plain = plan_slots(TAXONOMY, 30)
    with_edges = plan_slots(TAXONOMY, 30, edge_cases=("recovery", "conflicting_state"))

    assert [(s.index, s.category, s.difficulty) for s in plain] == [
        (s.index, s.category, s.difficulty) for s in with_edges
    ]


def test_requested_edge_cases_share_the_slots_with_plain_slots():
    slots = plan_slots(TAXONOMY, 60, edge_cases=("recovery", "conflicting_state"))
    counts = Counter(slot.edge_case for slot in slots)

    # Two types plus plain slots: about a third each.
    assert set(counts) == {"recovery", "conflicting_state", None}
    assert max(counts.values()) - min(counts.values()) <= 3


def test_edge_cases_spread_across_difficulty_levels():
    slots = plan_slots(TAXONOMY, 60, edge_cases=("recovery", "conflicting_state"))

    # 3 options and 5 levels share no rhythm with each other here, but even
    # with matching cycle lengths every level must see every option.
    for level in {1, 2, 3, 4, 5}:
        options = {s.edge_case for s in slots if s.difficulty == level}
        assert options == {"recovery", "conflicting_state", None}


def test_edge_cases_spread_when_the_cycle_matches_the_level_count():
    three_levels = _taxonomy(("triage", [1, 2, 3]), ("report", [1, 2, 3]))
    slots = plan_slots(three_levels, 36, edge_cases=("recovery", "boundary_conditions"))

    for level in (1, 2, 3):
        options = {s.edge_case for s in slots if s.difficulty == level}
        assert options == {"recovery", "boundary_conditions", None}


def test_an_unknown_edge_case_is_rejected():
    with pytest.raises(ValueError, match="sudo_errors"):
        plan_slots(TAXONOMY, 10, edge_cases=("sudo_errors",))


def test_a_repeated_edge_case_is_rejected():
    with pytest.raises(ValueError, match="recovery"):
        plan_slots(TAXONOMY, 10, edge_cases=("recovery", "recovery"))
