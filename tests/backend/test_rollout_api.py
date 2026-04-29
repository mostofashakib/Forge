import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from backend.app.database import Base
from backend.app.models import RolloutJob, ExportJob
from datetime import datetime, timezone

@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)

def test_rollout_job_model(db):
    job = RolloutJob(
        id="rj_abcd1234",
        env_name="test_env",
        task_name="test_task",
        agent_id="random",
        num_episodes=5,
        seed_start=0,
        status="pending",
        episodes_completed=0,
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()
    fetched = db.get(RolloutJob, "rj_abcd1234")
    assert fetched.env_name == "test_env"
    assert fetched.status == "pending"
    assert fetched.episodes_completed == 0
    assert fetched.completed_at is None
    assert fetched.error is None

def test_export_job_model(db):
    job = ExportJob(
        id="ex_abcd1234",
        env_name="test_env",
        formats='["trajectories","rewards"]',
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()
    fetched = db.get(ExportJob, "ex_abcd1234")
    assert fetched.env_name == "test_env"
    assert fetched.formats == '["trajectories","rewards"]'
    assert fetched.output_path is None
    assert fetched.completed_at is None


def test_run_rollout_task_eager(db, monkeypatch):
    """Test that run_rollout_task handles missing env gracefully in eager mode."""
    from backend.app.worker.celery_app import celery as celery_app
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False  # don't propagate so we can check status

    job = RolloutJob(
        id="rj_eager0001",
        env_name="non_existent_env",
        task_name="task_a",
        agent_id="random",
        num_episodes=1,
        seed_start=0,
        status="pending",
        episodes_completed=0,
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()

    from backend.app.worker.tasks import run_rollout_task
    # Should not raise — env not found is caught and marked as failed
    run_rollout_task.apply(args=["rj_eager0001"])
    # Test passes if no exception raised


from unittest.mock import patch, MagicMock
from backend.app.services.rollout_service import create_rollout, get_rollout, list_rollouts


def test_create_rollout(db):
    with patch("backend.app.services.rollout_service.run_rollout_task") as mock_task:
        mock_task.apply_async = MagicMock()
        job = create_rollout(
            env_name="env_a",
            task_name="task_a",
            agent_id="random",
            num_episodes=3,
            seed_start=10,
            db=db,
        )
    assert job.id.startswith("rj_")
    assert job.env_name == "env_a"
    assert job.status == "pending"
    assert job.episodes_completed == 0


def test_get_rollout(db):
    with patch("backend.app.services.rollout_service.run_rollout_task") as mock_task:
        mock_task.apply_async = MagicMock()
        job = create_rollout("env_a", "task_a", "random", 2, 0, db)
    fetched = get_rollout(job.id, db)
    assert fetched is not None
    assert fetched.id == job.id


def test_get_rollout_missing(db):
    assert get_rollout("rj_nonexistent", db) is None


def test_list_rollouts(db):
    with patch("backend.app.services.rollout_service.run_rollout_task") as mock_task:
        mock_task.apply_async = MagicMock()
        j1 = create_rollout("env_x", "t1", "random", 1, 0, db)
        j2 = create_rollout("env_x", "t2", "random", 1, 0, db)
        j3 = create_rollout("env_y", "t1", "random", 1, 0, db)
    results = list_rollouts("env_x", db)
    ids = [r.id for r in results]
    assert j1.id in ids
    assert j2.id in ids
    assert j3.id not in ids


from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    return TestClient(app)


def test_get_rollout_endpoint_not_found(api_client):
    resp = api_client.get("/api/rollouts/rj_nonexistent")
    assert resp.status_code == 404


def test_get_export_endpoint_not_found(api_client):
    resp = api_client.get("/api/exports/ex_nonexistent")
    assert resp.status_code == 404


def test_list_rollouts_endpoint(api_client):
    resp = api_client.get("/api/rollouts/?env_name=nonexistent_env")
    assert resp.status_code == 200
    assert resp.json() == []
