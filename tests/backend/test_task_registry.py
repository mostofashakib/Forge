"""The registry: versioned, dated, immutable batches with their taxonomy copy."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import GeneratedTask, TaskBatch, TaskRejectionRecord
from backend.app.services import task_registry
from forge.taskfactory.model_settings import ModelSpec
from forge.taskfactory.pipeline import PipelineResult
from forge.taskfactory.schemas import (
    Check,
    GoldenStep,
    ReviewVerdict,
    SyntheticTask,
    TaskRejection,
    TaskSeed,
    Taxonomy,
    TaxonomyCategory,
)

WRITER = ModelSpec(provider="anthropic", model="claude-sonnet-5")
VALIDATOR = ModelSpec(provider="openai", model="gpt-5")
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
        id=f"t-{n:04d}", category="triage", difficulty=1, title=f"task {n}", objective="o",
        seed=TaskSeed(), golden=[GoldenStep(tool="/a")],
        checks=[Check(kind="value", path="x", op="==", value=1)], reflection_points=[],
        step_budget=2, round=1, fingerprint="fp",
        review=ReviewVerdict(slot=n - 1, realistic=True, realistic_reason="r", fair=True,
                             fair_reason="f", sensible=True, sensible_reason="s"),
    )


def _result(requested: int, delivered: int) -> PipelineResult:
    rejections = [
        TaskRejection(slot=0, category="triage", difficulty=1, round=1, stage="static", reason="bad tool")
    ]
    return PipelineResult(
        requested=requested, taxonomy=TAXONOMY,
        tasks=[_task(n) for n in range(1, delivered + 1)], rejections=rejections,
    )


def _create(db, env="mail", requested=3) -> str:
    return task_registry.create_batch(db, env_name=env, requested=requested, pass_k=3, writer=WRITER, validator=VALIDATOR)


def test_a_new_batch_is_queued_without_a_version(db):
    batch = db.get(TaskBatch, _create(db))

    assert batch.status == "queued"
    assert batch.version is None
    assert batch.writer_model == "anthropic:claude-sonnet-5"
    assert batch.validator_model == "openai:gpt-5"


def test_saving_assigns_the_next_version_and_stores_everything(db):
    first, second = _create(db), _create(db)

    assert task_registry.save_result(db, first, _result(3, 3)) == 1
    assert task_registry.save_result(db, second, _result(3, 3)) == 2

    batch = db.get(TaskBatch, second)
    assert batch.status == "complete"
    assert batch.delivered == 3
    assert batch.completed_at is not None
    assert Taxonomy.model_validate_json(batch.taxonomy_json) == TAXONOMY
    assert db.query(GeneratedTask).filter_by(batch_id=second).count() == 3
    assert db.query(TaskRejectionRecord).filter_by(batch_id=second).count() == 1


def test_versions_count_per_environment(db):
    task_registry.save_result(db, _create(db, env="mail"), _result(1, 1))

    assert task_registry.save_result(db, _create(db, env="chat"), _result(1, 1)) == 1


def test_a_short_batch_is_saved_as_short(db):
    batch_id = _create(db, requested=5)

    task_registry.save_result(db, batch_id, _result(5, 3))

    assert db.get(TaskBatch, batch_id).status == "short"


def test_a_failed_batch_keeps_its_error_and_gets_no_version(db):
    batch_id = _create(db)

    task_registry.mark_failed(db, batch_id, "environment not running")

    batch = db.get(TaskBatch, batch_id)
    assert batch.status == "failed"
    assert batch.version is None
    assert batch.error == "environment not running"


def test_a_deleted_version_number_is_never_reused(db):
    first = _create(db)
    task_registry.save_result(db, first, _result(1, 1))
    task_registry.delete_batch(db, first)

    assert task_registry.save_result(db, _create(db), _result(1, 1)) == 2
    assert db.query(GeneratedTask).filter_by(batch_id=first).count() == 0


def test_listing_hides_deleted_batches_and_newest_comes_first(db):
    first, second = _create(db), _create(db)
    task_registry.save_result(db, first, _result(1, 1))
    task_registry.save_result(db, second, _result(1, 1))
    task_registry.delete_batch(db, first)

    assert [b["id"] for b in task_registry.list_batches(db, env_name="mail")] == [second]


def test_a_saved_batch_cannot_be_saved_again(db):
    batch_id = _create(db)
    task_registry.save_result(db, batch_id, _result(1, 1))

    with pytest.raises(ValueError, match="already"):
        task_registry.save_result(db, batch_id, _result(1, 1))


def test_the_detail_includes_tasks_rejections_and_taxonomy(db):
    batch_id = _create(db)
    task_registry.save_result(db, batch_id, _result(3, 2))

    detail = task_registry.get_batch(db, batch_id)

    assert [t["id"] for t in detail["tasks"]] == ["t-0001", "t-0002"]
    assert detail["rejections"][0]["reason"] == "bad tool"
    assert detail["taxonomy"]["categories"][0]["name"] == "triage"
    assert detail["shortfall"] == 1


def test_an_unknown_or_deleted_batch_has_no_detail(db):
    batch_id = _create(db)
    task_registry.delete_batch(db, batch_id)

    assert task_registry.get_batch(db, batch_id) is None
    assert task_registry.get_batch(db, "tb_missing") is None
