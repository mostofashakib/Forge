"""Calibration runs: one row per attempt, rates and summaries rebuilt from the rows."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import CalibrationAttempt, CalibrationRun, TaskBatch
from backend.app.services import calibration_registry, task_registry
from forge.taskfactory.attempt import Attempt
from forge.taskfactory.model_settings import ModelSpec
from forge.taskfactory.pipeline import PipelineResult
from forge.taskfactory.policy import Turn
from forge.taskfactory.runner import StepResult
from forge.taskfactory.schemas import (
    Check,
    GoldenStep,
    ReviewVerdict,
    SyntheticTask,
    TaskSeed,
    Taxonomy,
    TaxonomyCategory,
)
from forge.taskfactory.state_checks import CheckOutcome

TAXONOMY = Taxonomy(
    categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1, 2])],
    difficulty_rubric={"1": "one step"},
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _task(n: int, difficulty: int = 1) -> SyntheticTask:
    return SyntheticTask(
        id=f"t-{n:05d}", category="triage", difficulty=difficulty, title=f"task {n}", objective="archive",
        seed=TaskSeed(), golden=[GoldenStep(tool="/archive")],
        checks=[Check(kind="value", path="x", op="==", value=1, description="x is 1")],
        reflection_points=[], step_budget=2, round=1, fingerprint="fp",
        review=ReviewVerdict(slot=n - 1, realistic=True, realistic_reason="r", fair=True, fair_reason="f",
                             sensible=True, sensible_reason="s"),
    )


def _saved_batch(db, tasks) -> str:
    batch_id = task_registry.create_batch(
        db, env_name="mail", requested=len(tasks), pass_k=1,
        writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
    )
    task_registry.save_result(db, batch_id, PipelineResult(requested=len(tasks), taxonomy=TAXONOMY, tasks=tasks))
    return batch_id


def _attempt(passed: bool, termination: str = "submitted") -> Attempt:
    return Attempt(
        turns=(Turn(GoldenStep(tool="/archive", args={"id": "e1"}), StepResult(ok=True, output="done")),),
        checks=(CheckOutcome(passed, "" if passed else "x was 0"),),
        termination=termination,
    )


def test_a_new_run_is_queued_with_its_target_and_attempt_count(db):
    batch_id = _saved_batch(db, [_task(1)])

    run_id = calibration_registry.create_run(db, batch_id=batch_id, target_model="ollama:qwen3.8:27b", attempts=5)

    run = db.get(CalibrationRun, run_id)
    assert run.status == "queued"
    assert run.target_model == "ollama:qwen3.8:27b"
    assert run.attempts_per_task == 5


def test_each_attempt_is_stored_as_its_own_row(db):
    task = _task(1)
    run_id = calibration_registry.create_run(db, batch_id=_saved_batch(db, [task]), target_model="ollama:m", attempts=3)

    calibration_registry.record_attempts(db, run_id, task, [_attempt(True), _attempt(False), _attempt(False, "budget")])

    rows = db.query(CalibrationAttempt).filter_by(run_id=run_id).order_by(CalibrationAttempt.attempt_index).all()
    assert [(r.attempt_index, r.passed, r.termination) for r in rows] == [
        (0, True, "submitted"), (1, False, "submitted"), (2, False, "budget"),
    ]
    assert '"/archive"' in rows[0].steps_json
    assert "x was 0" in rows[1].checks_json


def test_the_detail_rebuilds_rates_bands_and_failure_summaries_from_the_rows(db):
    easy, mixed, hard = _task(1), _task(2, difficulty=2), _task(3, difficulty=2)
    run_id = calibration_registry.create_run(
        db, batch_id=_saved_batch(db, [easy, mixed, hard]), target_model="ollama:m", attempts=2,
    )
    calibration_registry.record_attempts(db, run_id, easy, [_attempt(True), _attempt(True)])
    calibration_registry.record_attempts(db, run_id, mixed, [_attempt(True), _attempt(False)])
    calibration_registry.record_attempts(db, run_id, hard, [_attempt(False), _attempt(False, "policy_error")])
    calibration_registry.complete_run(db, run_id)

    detail = calibration_registry.get_run(db, run_id)

    by_id = {t["task_id"]: t for t in detail["tasks"]}
    assert detail["status"] == "complete"
    assert by_id["t-00001"]["band"] == "too_easy"
    assert by_id["t-00001"]["failure_summary"] is None
    assert by_id["t-00002"]["solve_rate"] == 0.5
    assert by_id["t-00002"]["band"] == "frontier"
    assert by_id["t-00003"]["band"] == "unsolved"
    assert "2 of 2 attempts failed" in by_id["t-00003"]["failure_summary"]["text"]
    assert detail["summary"]["bands"] == {"too_easy": 1, "frontier": 1, "unsolved": 1, "suspect": 0}
    assert detail["summary"]["by_difficulty"]["2"] == {"too_easy": 0, "frontier": 1, "unsolved": 1, "suspect": 0}


def test_a_task_every_attempt_finished_while_missing_the_same_check_is_suspect(db):
    task = _task(1)
    run_id = calibration_registry.create_run(db, batch_id=_saved_batch(db, [task]), target_model="ollama:m", attempts=3)
    calibration_registry.record_attempts(db, run_id, task, [_attempt(False)] * 3)

    detail = calibration_registry.get_run(db, run_id)

    assert detail["tasks"][0]["band"] == "suspect"
    assert detail["tasks"][0]["failure_summary"]["suspect_checks"] == ["x is 1"]
    assert detail["summary"]["bands"]["suspect"] == 1
    assert detail["summary"]["bands"]["unsolved"] == 0


def test_a_tasks_attempts_come_back_in_order_with_their_steps(db):
    task = _task(1)
    run_id = calibration_registry.create_run(db, batch_id=_saved_batch(db, [task]), target_model="ollama:m", attempts=2)
    calibration_registry.record_attempts(db, run_id, task, [_attempt(False), _attempt(True)])

    attempts = calibration_registry.task_attempts(db, run_id, "t-00001")

    assert [a["passed"] for a in attempts] == [False, True]
    assert attempts[0]["steps"][0]["step"]["tool"] == "/archive"
    assert attempts[0]["labels"]["failed_checks"] == ["x is 1"]


def test_an_unknown_run_or_task_has_no_detail(db):
    assert calibration_registry.get_run(db, "cr_missing") is None
    assert calibration_registry.task_attempts(db, "cr_missing", "t-00001") == []


def test_two_models_on_one_batch_are_kept_apart_and_the_batch_is_unchanged(db):
    task = _task(1)
    batch_id = _saved_batch(db, [task])
    before = task_registry.get_batch(db, batch_id)
    first = calibration_registry.create_run(db, batch_id=batch_id, target_model="ollama:a", attempts=1)
    second = calibration_registry.create_run(db, batch_id=batch_id, target_model="ollama:b", attempts=1)
    calibration_registry.record_attempts(db, first, task, [_attempt(True)])
    calibration_registry.record_attempts(db, second, task, [_attempt(False)])

    assert calibration_registry.get_run(db, first)["tasks"][0]["band"] == "too_easy"
    assert calibration_registry.get_run(db, second)["tasks"][0]["band"] == "unsolved"
    assert [r["target_model"] for r in calibration_registry.list_runs(db, batch_id)] == ["ollama:b", "ollama:a"]
    assert task_registry.get_batch(db, batch_id) == before


def test_a_failed_run_keeps_its_error_and_the_attempts_it_recorded(db):
    task = _task(1)
    run_id = calibration_registry.create_run(db, batch_id=_saved_batch(db, [task]), target_model="ollama:m", attempts=1)
    calibration_registry.mark_running(db, run_id)
    calibration_registry.record_attempts(db, run_id, task, [_attempt(True)])

    calibration_registry.mark_failed(db, run_id, "ollama is down")

    detail = calibration_registry.get_run(db, run_id)
    assert detail["status"] == "failed"
    assert detail["error"] == "ollama is down"
    assert len(detail["tasks"]) == 1


def test_a_run_whose_worker_died_is_failed_by_the_reaper(db):
    run_id = calibration_registry.create_run(db, batch_id=_saved_batch(db, [_task(1)]), target_model="ollama:m", attempts=1)
    calibration_registry.mark_running(db, run_id)
    now = datetime(2026, 11, 12, 20, 0, 0)
    db.get(CalibrationRun, run_id).started_at = now - timedelta(hours=15)
    db.commit()

    reaped = calibration_registry.fail_abandoned_runs(db, older_than=timedelta(hours=14), now=now)

    assert reaped == [run_id]
    assert "worker" in db.get(CalibrationRun, run_id).error


def test_a_run_inside_the_limit_is_not_reaped(db):
    run_id = calibration_registry.create_run(db, batch_id=_saved_batch(db, [_task(1)]), target_model="ollama:m", attempts=1)
    calibration_registry.mark_running(db, run_id)

    assert calibration_registry.fail_abandoned_runs(db, older_than=timedelta(hours=14)) == []
    assert db.get(CalibrationRun, run_id).status == "running"
    assert db.query(TaskBatch).count() == 1
