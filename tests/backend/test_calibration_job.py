"""The calibration job: every task attempted, every attempt stored, failures recorded."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import CalibrationAttempt, CalibrationRun
from backend.app.services import calibration_registry, task_registry
from backend.app.services.task_factory_targets import Target, TargetUnavailable
from backend.app.worker import calibration
from forge.taskfactory.model_settings import ModelSpec
from forge.taskfactory.pipeline import PipelineResult
from forge.taskfactory.policy import Submit
from forge.taskfactory.profile import EnvironmentProfile, ToolInfo
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

PROFILE = EnvironmentProfile(
    env_name="mail", env_type="premade:gmail", family="state", tools=(ToolInfo(name="/archive"),),
)
TAXONOMY = Taxonomy(
    categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1])],
    difficulty_rubric={"1": "one step"},
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _task(n: int) -> SyntheticTask:
    return SyntheticTask(
        id=f"t-{n:05d}", category="triage", difficulty=1, title=f"task {n}", objective="archive",
        seed=TaskSeed(), golden=[GoldenStep(tool="/archive")],
        checks=[Check(kind="actions_called", tools=["/archive"])],
        reflection_points=[], step_budget=2, round=1, fingerprint="fp",
        review=ReviewVerdict(slot=n - 1, realistic=True, realistic_reason="r", fair=True, fair_reason="f",
                             sensible=True, sensible_reason="s"),
    )


class _Session:
    def __init__(self):
        self.archived = False

    def step(self, step):
        self.archived = self.archived or step.tool == "/archive"
        return StepResult(ok=True, output="ok")

    def evaluate(self, checks):
        return [CheckOutcome(self.archived) for _ in checks]

    def fingerprint(self):
        return "fp"


class _Runner:
    def __init__(self):
        self.batches = 0

    @contextmanager
    def batch(self):
        self.batches += 1
        yield

    @contextmanager
    def session(self, task):
        yield _Session()


class _Policy:
    """Archives on even attempts and gives up on odd ones."""

    def __init__(self, calls):
        self.calls = calls

    def next_step(self, history):
        if history:
            return Submit()
        attempt = len(self.calls)
        self.calls.append(attempt)
        return GoldenStep(tool="/archive") if attempt % 2 == 0 else Submit()


def _run(db, *, tasks=3, attempts=2, make_policy=None, target_error=None, events=None, locks=None,
         target_model="ollama:qwen3.8:27b", runner=None):
    batch_id = task_registry.create_batch(
        db, env_name="mail", requested=tasks, pass_k=1,
        writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
    )
    task_registry.save_result(
        db, batch_id, PipelineResult(requested=tasks, taxonomy=TAXONOMY, tasks=[_task(n) for n in range(1, tasks + 1)]),
    )
    run_id = calibration_registry.create_run(db, batch_id=batch_id, target_model=target_model, attempts=attempts)
    calls: list[int] = []
    runner = runner or _Runner()

    @contextmanager
    def open_target(_db, env_name, job_id):
        if target_error:
            raise target_error
        yield Target(runner=runner, profile=lambda: PROFILE)

    @contextmanager
    def exclusive(env_name):
        (locks if locks is not None else []).append(env_name)
        yield

    calibration.execute_calibration(
        db, run_id, publish=(events if events is not None else []).append, exclusive=exclusive,
        open_target=open_target, make_policy=make_policy or (lambda target, task, profile: _Policy(calls)),
    )
    return run_id


def test_every_task_gets_every_attempt_and_the_run_completes(db):
    events: list[dict] = []
    run_id = _run(db, tasks=3, attempts=2, events=events)

    rows = db.query(CalibrationAttempt).filter_by(run_id=run_id).all()
    assert len(rows) == 6
    assert db.get(CalibrationRun, run_id).status == "complete"
    assert sum(row.passed for row in rows) == 3
    assert events[-1]["done"] is True


def test_each_tasks_attempts_hold_the_environment_lock_once(db):
    locks: list[str] = []
    runner = _Runner()

    _run(db, tasks=3, attempts=2, locks=locks, runner=runner)

    assert locks == ["mail"] * 4  # the profile, then one per task
    assert runner.batches == 3


def test_the_policy_is_built_for_the_runs_target_model(db):
    seen: list[str] = []

    def make_policy(target, task, profile):
        seen.append(target)
        return _Policy([])

    _run(db, tasks=2, attempts=1, make_policy=make_policy, target_model="ollama:gemma4:26b")

    assert seen == ["ollama:gemma4:26b", "ollama:gemma4:26b"]


def test_a_crash_midway_fails_the_run_and_keeps_what_was_recorded(db):
    made: list[str] = []

    def make_policy(target, task, profile):
        made.append(task.id)
        if len(made) == 2:
            raise ConnectionError("ollama is down")
        return _Policy([])

    run_id = _run(db, tasks=3, attempts=2, make_policy=make_policy)

    run = db.get(CalibrationRun, run_id)
    assert run.status == "failed"
    assert "ollama is down" in run.error
    assert db.query(CalibrationAttempt).filter_by(run_id=run_id).count() == 2


def test_an_unavailable_environment_fails_the_run_with_its_reason(db):
    run_id = _run(db, target_error=TargetUnavailable("mail is stopped. Press Start."))

    run = db.get(CalibrationRun, run_id)
    assert run.status == "failed"
    assert "Press Start" in run.error


def test_an_unreachable_target_fails_the_run_before_any_attempt(db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    run_id = _run(db, target_model="openai:gpt-5")

    run = db.get(CalibrationRun, run_id)
    assert run.status == "failed"
    assert "OPENAI_API_KEY" in run.error
    assert db.query(CalibrationAttempt).filter_by(run_id=run_id).count() == 0


def test_a_run_past_its_time_limit_stops_between_tasks_and_keeps_its_attempts(db, monkeypatch):
    clock = iter([0, 10, calibration.CALIBRATION_TIME_LIMIT_S + 1])
    monkeypatch.setattr(calibration.time, "monotonic", lambda: next(clock))

    run_id = _run(db, tasks=3, attempts=2)

    run = db.get(CalibrationRun, run_id)
    assert run.status == "failed"
    assert "time limit" in run.error
    assert db.query(CalibrationAttempt).filter_by(run_id=run_id).count() == 2


def test_celerys_limits_fire_only_after_the_jobs_own_time_limit():
    job = calibration.run_calibration_task

    assert calibration.CALIBRATION_TIME_LIMIT_S < job.soft_time_limit < job.time_limit
    assert job.time_limit == calibration.CALIBRATION_HARD_LIMIT_S


def test_a_run_cut_off_by_the_soft_time_limit_is_failed_with_a_clear_reason(db):
    from celery.exceptions import SoftTimeLimitExceeded

    run_id = _run(db, target_error=SoftTimeLimitExceeded())

    assert "time limit" in db.get(CalibrationRun, run_id).error


def test_the_reaper_runs_on_a_schedule_and_when_the_worker_boots():
    from unittest.mock import patch

    from backend.app.worker import celery_app

    schedule = {entry["task"] for entry in celery_app.celery.conf.beat_schedule.values()}
    assert "backend.app.worker.calibration.reap_abandoned_calibrations_task" in schedule

    with patch("backend.app.worker.calibration.reap_abandoned_calibrations_task") as reap:
        celery_app._reap_abandoned_calibrations_on_boot()
    reap.delay.assert_called_once_with()
