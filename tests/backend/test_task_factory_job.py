"""The task factory job: models checked first, pipeline run, batch saved or failed."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import TaskBatch
from backend.app.services import task_registry
from backend.app.services.task_factory_targets import Target, TargetUnavailable
from backend.app.worker import task_factory
from forge.taskfactory.model_settings import ModelSpec
from forge.taskfactory.profile import EnvironmentProfile, ToolInfo
from forge.taskfactory.schemas import (
    Check,
    GoldenStep,
    TaskDraft,
    TaskDraftBatch,
    Taxonomy,
    TaxonomyCategory,
)
from forge.taskfactory.state_checks import CheckOutcome
from forge.taskfactory.runner import StepResult

ENV = {
    "FORGE_LLM_PROVIDER": "anthropic", "FORGE_LLM_MODEL_CAPABLE": "claude-sonnet-5",
    "FORGE_TASK_VALIDATOR_PROVIDER": "openai", "FORGE_TASK_VALIDATOR_MODEL": "gpt-5",
}
PROFILE = EnvironmentProfile(
    env_name="mail", env_type="premade:gmail", family="state",
    tools=(ToolInfo(name="/archive_email"),), state_sample={"emails": []},
)
TAXONOMY = Taxonomy(
    categories=[
        TaxonomyCategory(name=n, description=d, exercises=["/archive_email"], difficulties=levels)
        for n, d, levels in [
            ("triage", "sort new mail by urgency", [1]),
            ("cleanup", "archive stale newsletters", [1, 2]),
            ("follow-up", "chase unanswered threads", [2, 3]),
            ("audit", "verify labels against policy", [3]),
        ]
    ],
    difficulty_rubric={"1": "one step"},
)


@pytest.fixture(autouse=True)
def sdks_installed(monkeypatch):
    import forge.taskfactory.model_settings as settings

    monkeypatch.setattr(settings, "_installed", lambda module: True)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


class _Client:
    """Writer and validator in one scripted client: answers by schema."""

    def __init__(self, label: str) -> None:
        self.label = label

    def extract(self, system, user, schema):
        if schema is Taxonomy:
            return TAXONOMY
        slots = [int(line.split()[1]) for line in user.splitlines() if line.startswith("SLOT ")]
        if schema is TaskDraftBatch:
            return TaskDraftBatch(tasks=[
                TaskDraft(
                    slot=s, title=f"task {s}", objective="archive",
                    golden=[GoldenStep(tool="/archive_email")] * {1: 1, 2: 5, 3: 12}[_level(s)],
                    checks=[Check(kind="record_exists", collection="emails", match={"id": "e1"}, expect={"folder": "archive"})],
                )
                for s in slots
            ])
        tasks = [int(line.split()[2]) for line in user.splitlines() if line.startswith("TASK slot ")]
        return schema.model_validate({"verdicts": [
            {"slot": s, "realistic": True, "realistic_reason": "r", "fair": True, "fair_reason": "f",
             "sensible": True, "sensible_reason": "s", "edge_case_ok": True, "edge_case_reason": "planted",
             "check_traces": [{"check": 0, "requires": "e1 archived", "asked_by": "archive"}]}
            for s in tasks
        ]})


def _level(slot: int) -> int:
    return [1, 2, 3][slot % 3]


class _Session:
    def __init__(self):
        self.steps = 0

    def step(self, step):
        self.steps += 1
        return StepResult(ok=True)

    def evaluate(self, checks):
        return [CheckOutcome(self.steps > 0) for _ in checks]

    def fingerprint(self):
        return "fp"


class _Runner:
    @contextmanager
    def batch(self):
        yield

    @contextmanager
    def session(self, draft):
        yield _Session()


def _job(db, *, env=ENV, target_error=None, events=None, locks=None):
    clients: list[ModelSpec] = []

    def make_client(spec: ModelSpec, max_tokens: int):
        clients.append(spec)
        return _Client(spec.model)

    @contextmanager
    def open_target(_db, env_name, batch_id):
        if target_error:
            raise target_error
        yield Target(runner=_Runner(), profile=lambda: PROFILE)

    @contextmanager
    def exclusive(env_name):
        (locks if locks is not None else []).append(env_name)
        yield

    batch_id = task_registry.create_batch(
        db, env_name="mail", requested=3, pass_k=2,
        writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
    )
    task_factory.execute_batch(
        db, batch_id, environ=env, publish=(events if events is not None else []).append,
        make_client=make_client, open_target=open_target, exclusive=exclusive,
    )
    return batch_id, clients


def test_a_batch_runs_end_to_end_and_is_saved_as_version_one(db):
    events: list[dict] = []
    batch_id, clients = _job(db, events=events)

    batch = db.get(TaskBatch, batch_id)
    assert batch.status == "complete", batch.error
    assert batch.version == 1
    assert batch.delivered == 3
    assert [c.model for c in clients] == ["claude-sonnet-5", "gpt-5"]
    assert events[-1]["done"] is True
    assert events[-1]["version"] == 1


def test_the_profile_and_each_round_hold_the_environment_lock(db):
    locks: list[str] = []

    _job(db, locks=locks)

    assert locks == ["mail", "mail"]


def test_a_same_family_validator_fails_the_batch_before_any_llm_call(db):
    env = {**ENV, "FORGE_TASK_VALIDATOR_PROVIDER": "anthropic", "FORGE_TASK_VALIDATOR_MODEL": "claude-haiku-4-5"}
    events: list[dict] = []

    batch_id, clients = _job(db, env=env, events=events)

    batch = db.get(TaskBatch, batch_id)
    assert batch.status == "failed"
    assert "FORGE_TASK_VALIDATOR_MODEL" in batch.error
    assert clients == []
    assert "FORGE_TASK_VALIDATOR_MODEL" in events[-1]["error"]


def test_a_missing_validator_sdk_fails_the_batch_before_any_llm_call(db, monkeypatch):
    import forge.taskfactory.model_settings as settings

    monkeypatch.setattr(settings, "_installed", lambda module: module != "openai")

    batch_id, clients = _job(db)

    assert clients == []
    assert "openai" in db.get(TaskBatch, batch_id).error


def test_an_unavailable_environment_fails_the_batch_with_its_reason(db):
    batch_id, _ = _job(db, target_error=TargetUnavailable("mail is stopped. Start it, then create tasks."))

    batch = db.get(TaskBatch, batch_id)
    assert batch.status == "failed"
    assert "Start it" in batch.error
    assert batch.version is None


def test_the_batch_records_the_models_that_actually_ran(db):
    env = {**ENV, "FORGE_TASK_VALIDATOR_PROVIDER": "gemini", "FORGE_TASK_VALIDATOR_MODEL": "gemini-3-pro"}

    batch_id, _ = _job(db, env=env)

    assert db.get(TaskBatch, batch_id).validator_model == "gemini:gemini-3-pro"


class _RecordingClient(_Client):
    def __init__(self, label: str, prompts: list[str]) -> None:
        super().__init__(label)
        self._prompts = prompts

    def extract(self, system, user, schema):
        if schema is TaskDraftBatch:
            self._prompts.append(user)
        return super().extract(system, user, schema)


def test_the_batchs_edge_cases_reach_the_writer(db):
    prompts: list[str] = []

    @contextmanager
    def open_target(_db, env_name, batch_id):
        yield Target(runner=_Runner(), profile=lambda: PROFILE)

    @contextmanager
    def exclusive(env_name):
        yield

    batch_id = task_registry.create_batch(
        db, env_name="mail", requested=3, pass_k=1,
        writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
        edge_cases=["recovery"],
    )
    task_factory.execute_batch(
        db, batch_id, environ=ENV, publish=lambda _event: None,
        make_client=lambda spec, max_tokens: _RecordingClient(spec.model, prompts),
        open_target=open_target, exclusive=exclusive,
    )

    assert db.get(TaskBatch, batch_id).status != "failed", db.get(TaskBatch, batch_id).error
    assert any("edge case recovery" in prompt for prompt in prompts)


# ---------------------------------------------------------------------------
# Time limits: the pipeline stops itself first, Celery's limits back it up,
# and a batch whose worker died is failed by the reaper.
# ---------------------------------------------------------------------------

def test_the_pipeline_runs_under_the_batch_time_limit(db, monkeypatch):
    seen: dict = {}
    real = task_factory.TaskFactoryPipeline

    def recording(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(task_factory, "TaskFactoryPipeline", recording)

    _job(db)

    assert seen["time_limit_s"] == task_factory.BATCH_TIME_LIMIT_S


def test_celerys_limits_fire_only_after_the_pipelines_own_time_limit():
    job = task_factory.create_task_batch_task

    assert task_factory.BATCH_TIME_LIMIT_S < job.soft_time_limit < job.time_limit
    assert job.time_limit == task_factory.BATCH_HARD_LIMIT_S


def test_a_batch_cut_off_by_the_soft_time_limit_is_failed_with_a_clear_reason(db):
    from celery.exceptions import SoftTimeLimitExceeded

    events: list[dict] = []
    batch_id, _ = _job(db, target_error=SoftTimeLimitExceeded(), events=events)

    batch = db.get(TaskBatch, batch_id)
    assert batch.status == "failed"
    assert "time limit" in batch.error
    assert "time limit" in events[-1]["error"]


def test_the_reaper_fails_only_batches_past_the_hard_limit(db):
    from datetime import datetime, timedelta, timezone

    def running(started):
        batch_id = task_registry.create_batch(
            db, env_name="mail", requested=1, pass_k=1,
            writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
        )
        db.get(TaskBatch, batch_id).status = "running"
        db.get(TaskBatch, batch_id).started_at = started
        db.commit()
        return batch_id

    now = datetime.now(timezone.utc)
    dead = running(now - timedelta(seconds=task_factory.BATCH_HARD_LIMIT_S + 60))
    alive = running(now - timedelta(seconds=task_factory.BATCH_HARD_LIMIT_S - 60))

    assert task_factory.reap_abandoned_batches(db) == [dead]
    assert db.get(TaskBatch, alive).status == "running"


def test_the_reaper_runs_on_a_schedule_and_when_the_worker_boots(monkeypatch):
    from unittest.mock import patch

    from backend.app.worker import celery_app

    schedule = {entry["task"] for entry in celery_app.celery.conf.beat_schedule.values()}
    assert "backend.app.worker.task_factory.reap_abandoned_batches_task" in schedule

    with patch("backend.app.worker.task_factory.reap_abandoned_batches_task") as reap:
        celery_app._reap_abandoned_batches_on_boot()
    reap.delay.assert_called_once_with()


def test_the_boot_reaper_retries_while_the_api_has_not_migrated_the_schema_yet():
    # The worker can boot before the API's init_db adds a new column.
    from sqlalchemy.exc import OperationalError

    reaper = task_factory.reap_abandoned_batches_task

    assert OperationalError in reaper.autoretry_for
    assert reaper.max_retries * reaper.default_retry_delay >= 5 * 60
