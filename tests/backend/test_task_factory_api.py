"""The task factory API: pick an environment and a count, then browse versions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import SandboxEnvironment
from backend.app.services import env_file, task_registry
from forge.taskfactory.model_settings import ModelSpec
from forge.taskfactory.pipeline import PipelineResult
from forge.taskfactory.schemas import Taxonomy, TaxonomyCategory


@pytest.fixture(autouse=True)
def sdks_installed(monkeypatch):
    import forge.taskfactory.model_settings as settings

    monkeypatch.setattr(settings, "_installed", lambda module: True)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-5")
    env_path = tmp_path / ".env"
    env_path.write_text("FORGE_TASK_VALIDATOR_PROVIDER=openai\nFORGE_TASK_VALIDATOR_MODEL=gpt-5\n")
    monkeypatch.setattr(env_file, "BACKEND_ENV_FILE", env_path)
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    with database.get_session_factory()() as db:
        for name, status in (("mail", "running"), ("web", "stopped")):
            db.add(SandboxEnvironment(
                id=name, status=status, env_type="premade:gmail" if name == "mail" else "browser",
                container_id="c1", expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            ))
        db.commit()
    client = TestClient(app)
    client.env_path = env_path
    client.session_factory = database.get_session_factory()
    return client


@pytest.fixture
def queued():
    with patch("backend.app.api.task_factory.create_task_batch_task") as task:
        task.delay = MagicMock()
        yield task


def test_environments_report_type_and_readiness(client):
    body = {e["name"]: e for e in client.get("/api/task-factory/environments").json()}

    assert body["mail"]["family"] == "state" and body["mail"]["ready"] is True
    assert body["web"]["ready"] is False


def test_starting_a_batch_queues_the_job(client, queued):
    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 12, "k": 3})

    assert resp.status_code == 202
    batch_id = resp.json()["batch_id"]
    queued.delay.assert_called_once_with(batch_id=batch_id)
    batch = client.get(f"/api/task-factory/batches/{batch_id}").json()
    assert batch["status"] == "queued"
    assert batch["requested"] == 12 and batch["pass_k"] == 3
    assert batch["validator_model"] == "openai:gpt-5"


def test_a_batch_of_twenty_thousand_is_accepted(client, queued):
    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 20_000, "k": 3})

    assert resp.status_code == 202


def test_a_stopped_environment_is_refused(client, queued):
    resp = client.post("/api/task-factory/batches", json={"env_name": "web", "count": 5, "k": 3})

    assert resp.status_code == 409
    assert "Start" in resp.json()["detail"]
    queued.delay.assert_not_called()


def test_an_unknown_environment_is_refused(client, queued):
    resp = client.post("/api/task-factory/batches", json={"env_name": "nope", "count": 5, "k": 3})

    assert resp.status_code == 404


def test_missing_validator_settings_are_refused_by_name(client, queued):
    client.env_path.write_text("")

    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 5, "k": 3})

    assert resp.status_code == 422
    assert "FORGE_TASK_VALIDATOR_PROVIDER" in resp.json()["detail"]


def test_a_validator_sdk_that_is_not_installed_is_refused_before_queueing(client, queued, monkeypatch):
    import forge.taskfactory.model_settings as settings

    monkeypatch.setattr(settings, "_installed", lambda module: module != "openai")

    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 5, "k": 3})

    assert resp.status_code == 422
    assert "openai" in resp.json()["detail"]
    queued.delay.assert_not_called()


@pytest.mark.parametrize("count, k", [(0, 3), (20_001, 3), (5, 0), (5, 11)])
def test_count_and_k_outside_their_limits_are_refused(client, queued, count, k):
    resp = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": count, "k": k})

    assert resp.status_code == 422


def _saved_batch(client) -> str:
    taxonomy = Taxonomy(
        categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1])],
    )
    with client.session_factory() as db:
        batch_id = task_registry.create_batch(
            db, env_name="mail", requested=2, pass_k=3,
            writer=ModelSpec("anthropic", "claude-sonnet-5"), validator=ModelSpec("openai", "gpt-5"),
        )
        task_registry.save_result(db, batch_id, PipelineResult(requested=2, taxonomy=taxonomy))
    return batch_id


def test_batches_are_listed_per_environment(client):
    batch_id = _saved_batch(client)

    listed = client.get("/api/task-factory/batches", params={"env_name": "mail"}).json()

    assert [b["id"] for b in listed] == [batch_id]
    assert client.get("/api/task-factory/batches", params={"env_name": "web"}).json() == []


def test_a_saved_batch_exports_as_a_dated_json_file(client):
    batch_id = _saved_batch(client)

    resp = client.get(f"/api/task-factory/batches/{batch_id}/export")

    assert resp.status_code == 200
    disposition = resp.headers["content-disposition"]
    assert "mail-" in disposition and disposition.endswith('.json"')
    body = resp.json()
    assert body["version"] == 1
    assert body["taxonomy"]["categories"][0]["name"] == "triage"
    assert body["created_at"]


def test_an_unsaved_batch_cannot_be_exported(client, queued):
    batch_id = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 1, "k": 1}).json()["batch_id"]

    assert client.get(f"/api/task-factory/batches/{batch_id}/export").status_code == 409


def test_a_saved_batch_can_be_deleted(client):
    batch_id = _saved_batch(client)

    assert client.delete(f"/api/task-factory/batches/{batch_id}").status_code == 204
    assert client.get(f"/api/task-factory/batches/{batch_id}").status_code == 404


def test_a_running_batch_cannot_be_deleted(client, queued):
    batch_id = client.post("/api/task-factory/batches", json={"env_name": "mail", "count": 1, "k": 1}).json()["batch_id"]

    assert client.delete(f"/api/task-factory/batches/{batch_id}").status_code == 409


def test_an_unknown_batch_is_not_found(client):
    assert client.get("/api/task-factory/batches/tb_missing").status_code == 404
    assert client.delete("/api/task-factory/batches/tb_missing").status_code == 404
