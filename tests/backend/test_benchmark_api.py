import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from backend.app.database import Base
from backend.app.models import BenchmarkRun
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_benchmark_run_model(db):
    run = BenchmarkRun(
        id="bm_test0001",
        status="queued",
        domains="email,project_mgmt",
        depth=5,
        seeds=5,
        output_dir="benchmark_results",
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    fetched = db.get(BenchmarkRun, "bm_test0001")
    assert fetched.status == "queued"
    assert fetched.domains == "email,project_mgmt"
    assert fetched.completed_at is None
    assert fetched.error is None
    assert fetched.report_json is None


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    return TestClient(app)


def test_list_benchmark_runs_empty(api_client):
    resp = api_client.get("/api/benchmark/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_benchmark_run_not_found(api_client):
    resp = api_client.get("/api/benchmark/runs/bm_nonexistent")
    assert resp.status_code == 404


def test_create_benchmark_run(api_client):
    with patch("backend.app.api.benchmark.run_benchmark_task") as mock_task:
        mock_task.delay = MagicMock(return_value=MagicMock(id="celery-id-1"))
        resp = api_client.post("/api/benchmark/runs", json={
            "domains": ["email"],
            "depth": 3,
            "seeds": 2,
            "output_dir": "bench_out",
        })
    assert resp.status_code == 202
    data = resp.json()
    assert "run_id" in data
    assert data["run_id"].startswith("bm_")


def test_get_benchmark_run_report_not_ready(api_client):
    with patch("backend.app.api.benchmark.run_benchmark_task") as mock_task:
        mock_task.delay = MagicMock(return_value=MagicMock(id="celery-id-2"))
        create_resp = api_client.post("/api/benchmark/runs", json={
            "domains": ["email"],
            "depth": 2,
            "seeds": 1,
            "output_dir": "bench_out2",
        })
    run_id = create_resp.json()["run_id"]
    resp = api_client.get(f"/api/benchmark/runs/{run_id}/report")
    assert resp.status_code == 404


def test_get_benchmark_run_report_with_data(api_client, tmp_path):
    import json
    from backend.app.database import get_session_factory
    from backend.app.models import BenchmarkRun
    from datetime import datetime, timezone

    report_data = [{"env_name": "email", "state_coverage_score": 0.8,
                    "reward_density": 0.5, "dead_end_rate": 0.1,
                    "action_diversity": 0.7, "num_episodes": 10, "num_steps": 100}]
    SessionLocal = get_session_factory()
    with SessionLocal() as db:
        run = BenchmarkRun(
            id="bm_withreport",
            status="done",
            domains="email",
            depth=3,
            seeds=2,
            output_dir=str(tmp_path),
            created_at=datetime.now(timezone.utc),
            report_json=json.dumps(report_data),
        )
        db.add(run)
        db.commit()

    resp = api_client.get("/api/benchmark/runs/bm_withreport/report")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["env_name"] == "email"
    assert body[0]["state_coverage_score"] == 0.8


def test_run_benchmark_task_no_redis(monkeypatch, tmp_path):
    """Task fails gracefully when Redis is unreachable."""
    from backend.app.worker.celery_app import celery as celery_app
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False

    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/task_test.db")
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()

    from backend.app.database import get_session_factory
    from backend.app.models import BenchmarkRun
    SessionLocal = get_session_factory()
    with SessionLocal() as db:
        db.add(BenchmarkRun(
            id="bm_tasktest01",
            status="queued",
            domains="email",
            depth=1,
            seeds=1,
            output_dir=str(tmp_path / "out"),
            created_at=datetime.now(timezone.utc),
        ))
        db.commit()

    monkeypatch.setenv("REDIS_URL", "redis://localhost:19999/0")  # unreachable port
    from backend.app.worker.tasks import run_benchmark_task
    run_benchmark_task.apply(args=["bm_tasktest01", ["email"], 1, 1, str(tmp_path / "out")])

    with SessionLocal() as db:
        run = db.get(BenchmarkRun, "bm_tasktest01")
        assert run.status == "failed"
        assert run.error is not None
