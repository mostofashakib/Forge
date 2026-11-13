"""Solve rates, bands, and the summary of one calibration run."""
from __future__ import annotations

import pytest

from forge.taskfactory.calibration import (
    MAX_ATTEMPTS,
    TaskRate,
    attempt_count_problem,
    band,
    calibration_summary,
)


def test_a_task_never_solved_is_unsolved():
    assert band(passed=0, attempts=5) == "unsolved"


def test_a_task_always_solved_is_too_easy():
    assert band(passed=5, attempts=5) == "too_easy"


@pytest.mark.parametrize("passed", [1, 2, 3, 4])
def test_a_task_solved_sometimes_is_on_the_frontier(passed):
    assert band(passed=passed, attempts=5) == "frontier"


@pytest.mark.parametrize("passed, attempts", [(0, 0), (6, 5), (-1, 5)])
def test_impossible_counts_are_rejected(passed, attempts):
    with pytest.raises(ValueError):
        band(passed=passed, attempts=attempts)


def test_attempts_are_capped_at_five():
    assert MAX_ATTEMPTS == 5
    assert attempt_count_problem(5) is None
    assert attempt_count_problem(1) is None
    assert "1 to 5" in attempt_count_problem(6)
    assert "1 to 5" in attempt_count_problem(0)


def _rate(task_id, category, difficulty, passed, attempts=5):
    return TaskRate(task_id=task_id, category=category, difficulty=difficulty, passed=passed, attempts=attempts)


def test_the_summary_counts_bands_overall_by_category_and_by_difficulty():
    rates = [
        _rate("t1", "triage", 1, 5),
        _rate("t2", "triage", 1, 3),
        _rate("t3", "cleanup", 3, 0),
        _rate("t4", "cleanup", 3, 1),
    ]

    summary = calibration_summary(rates)

    assert summary["tasks"] == 4
    assert summary["bands"] == {"too_easy": 1, "frontier": 2, "unsolved": 1, "suspect": 0}
    assert summary["by_category"]["triage"] == {"too_easy": 1, "frontier": 1, "unsolved": 0, "suspect": 0}
    assert summary["by_difficulty"]["3"] == {"too_easy": 0, "frontier": 1, "unsolved": 1, "suspect": 0}
    assert summary["histogram"] == {"0.0": 1, "0.2": 1, "0.4": 0, "0.6": 1, "0.8": 0, "1.0": 1}


def test_an_empty_run_has_an_empty_summary():
    summary = calibration_summary([])

    assert summary["tasks"] == 0
    assert summary["bands"] == {"too_easy": 0, "frontier": 0, "unsolved": 0, "suspect": 0}
    assert summary["histogram"] == {}


def test_runs_with_different_attempt_counts_are_not_mixed_in_one_histogram():
    with pytest.raises(ValueError, match="attempt"):
        calibration_summary([_rate("t1", "c", 1, 1, attempts=5), _rate("t2", "c", 1, 1, attempts=3)])


def test_an_unsolved_task_whose_attempts_all_missed_the_same_check_is_suspect_not_unsolved():
    rates = [
        TaskRate(task_id="t1", category="ops", difficulty=2, passed=0, attempts=5, suspect=True),
        _rate("t2", "ops", 2, 0),
    ]

    summary = calibration_summary(rates)

    assert rates[0].band == "suspect"
    assert summary["bands"] == {"too_easy": 0, "frontier": 0, "unsolved": 1, "suspect": 1}
    assert summary["histogram"]["0.0"] == 2


def test_a_task_some_attempt_solved_cannot_be_suspect():
    with pytest.raises(ValueError):
        TaskRate(task_id="t1", category="ops", difficulty=2, passed=1, attempts=5, suspect=True)
