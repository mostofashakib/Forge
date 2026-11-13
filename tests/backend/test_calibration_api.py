"""The calibration API: start a run on a saved batch, then read rates and attempts."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import calibration_registry, task_registry
from forge.taskfactory.attempt import Attempt
from forge.taskfactory.model_settings import ModelSpec
from forge.taskfactory.pipeline import PipelineResult
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
from tests.backend.safety_helpers import use_safety_env

TAXONOMY = Taxonomy(
    categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1])],
    difficulty_rubric={"1": "one step"},
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    use_safety_env(monkeypatch)
    from backend.app import database

    database._engine = None
    database._SessionLocal = None
    database.init_db()
    client = TestClient(app)
    client.session_factory = database.get_session_factory()
    return client


@pytest.fixture
def queued():
    with patch("backend.app.api.calibration.run_calibration_task") as task:
        task.delay = MagicMock()
        yield task


def _task(n: int) -> SyntheticTask:
    return SyntheticTask(
        id=f"t-{n:05d}", category="triage", difficulty=1, title=f"task {n}", objective="archive",
        seed=TaskSeed(), golden=[GoldenStep(tool="/archive")],
        checks=[Check(kind="actions_called", tools=["/archive"], description="archived")],
        reflection_points=[], step_budget=2, round=1, fingerprint="fp",
        review=ReviewVerdict(slot=n - 1, realistic=True, realistic_reason="r", fair=True, fair_reason="f",
                             sensible=True, sensible_reason="s"),
    )


def _batch(client, *, saved: bool = True, tasks: int = 2) -> str:
    with client.session_factory() as db:
        batch_id = task_registry.create_batch(
            db, env_name="mail", requested=max(tasks, 1), pass_k=1,
            writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
        )
        if saved:
            task_registry.save_result(
                db, batch_id,
                PipelineResult(requested=max(tasks, 1), taxonomy=TAXONOMY, tasks=[_task(n) for n in range(1, tasks + 1)]),
            )
    return batch_id


def _url(batch_id: str) -> str:
    return f"/api/task-factory/batches/{batch_id}/calibrations"


def test_starting_a_calibration_queues_the_job_with_five_attempts_by_default(client, queued):
    batch_id = _batch(client)

    resp = client.post(_url(batch_id), json={"target_model": "ollama:qwen3.8:27b"})

    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    queued.delay.assert_called_once_with(run_id=run_id)
    run = client.get(f"{_url(batch_id)}/{run_id}").json()
    assert run["status"] == "queued"
    assert run["target_model"] == "ollama:qwen3.8:27b"
    assert run["attempts_per_task"] == 5


@pytest.mark.parametrize("attempts", [0, 6, 8])
def test_attempts_outside_one_to_five_are_refused(client, queued, attempts):
    resp = client.post(_url(_batch(client)), json={"target_model": "ollama:m", "attempts": attempts})

    assert resp.status_code == 422
    queued.delay.assert_not_called()


def test_a_hosted_target_without_its_key_is_refused_and_names_the_setting(client, queued):
    resp = client.post(_url(_batch(client)), json={"target_model": "openai:gpt-5"})

    assert resp.status_code == 422
    assert "OPENAI_API_KEY" in resp.json()["detail"]
    queued.delay.assert_not_called()


@pytest.mark.parametrize("target", ["gemini:x", "random", "ollama:"])
def test_an_unusable_target_is_refused(client, queued, target):
    resp = client.post(_url(_batch(client)), json={"target_model": target})

    assert resp.status_code == 422
    queued.delay.assert_not_called()


def test_an_unknown_batch_is_not_found(client, queued):
    assert client.post(_url("tb_missing"), json={"target_model": "ollama:m"}).status_code == 404
    assert client.get(_url("tb_missing")).status_code == 404


def test_an_unsaved_or_empty_batch_cannot_be_calibrated(client, queued):
    unsaved = client.post(_url(_batch(client, saved=False)), json={"target_model": "ollama:m"})
    empty = client.post(_url(_batch(client, tasks=0)), json={"target_model": "ollama:m"})

    assert unsaved.status_code == 409
    assert empty.status_code == 409
    queued.delay.assert_not_called()


def test_runs_are_listed_newest_first(client, queued):
    batch_id = _batch(client)
    first = client.post(_url(batch_id), json={"target_model": "ollama:a"}).json()["run_id"]
    second = client.post(_url(batch_id), json={"target_model": "ollama:b"}).json()["run_id"]

    assert [r["id"] for r in client.get(_url(batch_id)).json()] == [second, first]


def test_the_detail_has_rates_and_a_tasks_attempts_can_be_opened(client, queued):
    batch_id = _batch(client, tasks=1)
    run_id = client.post(_url(batch_id), json={"target_model": "ollama:m", "attempts": 2}).json()["run_id"]
    with client.session_factory() as db:
        calibration_registry.record_attempts(db, run_id, _task(1), [
            Attempt(turns=(), checks=(CheckOutcome(True),), termination="submitted"),
            Attempt(turns=(), checks=(CheckOutcome(False),), termination="budget"),
        ])

    detail = client.get(f"{_url(batch_id)}/{run_id}").json()
    attempts = client.get(f"{_url(batch_id)}/{run_id}/tasks/t-00001").json()

    assert detail["tasks"][0]["band"] == "frontier"
    assert detail["summary"]["bands"]["frontier"] == 1
    assert [a["termination"] for a in attempts] == ["submitted", "budget"]


def test_a_run_read_through_another_batch_is_not_found(client, queued):
    run_id = client.post(_url(_batch(client)), json={"target_model": "ollama:m"}).json()["run_id"]
    other = _batch(client)

    assert client.get(f"{_url(other)}/{run_id}").status_code == 404
    assert client.get(f"{_url(other)}/{run_id}/tasks/t-00001").status_code == 404
