"""A person's judgment on a calibrated task, and which tasks it keeps out of exports."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.services import calibration_registry, task_judgments
from tests.backend.test_calibration_registry import _attempt, _saved_batch, _task


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _calibrated(db, attempts_by_task: dict[int, list]) -> str:
    tasks = {n: _task(n) for n in attempts_by_task}
    batch_id = _saved_batch(db, list(tasks.values()))
    run_id = calibration_registry.create_run(db, batch_id=batch_id, target_model="ollama:m", attempts=3)
    for n, attempts in attempts_by_task.items():
        calibration_registry.record_attempts(db, run_id, tasks[n], attempts)
    return batch_id


SUSPECT = [_attempt(False)] * 3


def test_a_judgment_is_stored_and_a_later_one_replaces_it(db):
    batch_id = _saved_batch(db, [_task(1)])

    task_judgments.judge(db, batch_id, "t-00001", "unfair")
    task_judgments.judge(db, batch_id, "t-00001", "hard")

    assert task_judgments.judgments(db, batch_id) == {"t-00001": "hard"}


def test_a_verdict_other_than_hard_or_unfair_is_refused(db):
    batch_id = _saved_batch(db, [_task(1)])

    with pytest.raises(ValueError, match="hard or unfair"):
        task_judgments.judge(db, batch_id, "t-00001", "fine")
    assert task_judgments.judgments(db, batch_id) == {}


def test_a_task_outside_the_batch_cannot_be_judged(db):
    batch_id = _saved_batch(db, [_task(1)])

    with pytest.raises(LookupError):
        task_judgments.judge(db, batch_id, "t-00099", "hard")
    with pytest.raises(LookupError):
        task_judgments.judge(db, "tb_missing", "t-00001", "hard")


def test_judgments_belong_to_their_own_batch(db):
    first = _saved_batch(db, [_task(1)])
    second = _saved_batch(db, [_task(1)])

    task_judgments.judge(db, first, "t-00001", "unfair")

    assert task_judgments.judgments(db, second) == {}
    assert task_judgments.held_out(db, second) == {}


def test_a_task_marked_unfair_is_held_out(db):
    batch_id = _saved_batch(db, [_task(1), _task(2)])

    task_judgments.judge(db, batch_id, "t-00001", "unfair")

    assert list(task_judgments.held_out(db, batch_id)) == ["t-00001"]
    assert "unfair" in task_judgments.held_out(db, batch_id)["t-00001"]


def test_a_suspect_task_is_held_out_until_a_person_marks_it_hard(db):
    batch_id = _calibrated(db, {1: SUSPECT})

    reason = task_judgments.held_out(db, batch_id)["t-00001"]
    assert "suspect" in reason and "x is 1" in reason

    task_judgments.judge(db, batch_id, "t-00001", "hard")
    assert task_judgments.held_out(db, batch_id) == {}


def test_tasks_calibration_did_not_flag_stay_in(db):
    unsolved_by_budget = [_attempt(False, termination="budget")] * 3
    batch_id = _calibrated(db, {1: [_attempt(True)] * 3, 2: [_attempt(True), _attempt(False), _attempt(False)],
                                3: unsolved_by_budget})

    assert task_judgments.held_out(db, batch_id) == {}


def test_a_hard_mark_does_not_release_a_task_also_marked_unfair_later(db):
    batch_id = _calibrated(db, {1: SUSPECT})

    task_judgments.judge(db, batch_id, "t-00001", "hard")
    task_judgments.judge(db, batch_id, "t-00001", "unfair")

    assert "unfair" in task_judgments.held_out(db, batch_id)["t-00001"]
