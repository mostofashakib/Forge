"""Tests for the Training API router (/api/training)."""
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import TrainingRun
from backend.app.database import get_session_factory, init_db
from forge.training.checkpoint import PolicyCheckpoint


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test_training.db")
    monkeypatch.chdir(tmp_path)
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    init_db()
    return TestClient(app)


def test_create_training_run_invalid_confined_path_fails(client):
    res = client.post(
        "/api/training/runs",
        json={
            "base_model": "test-model",
            "data_dir": "../../outside_root_dir",
            "objective": "grpo",
        },
    )
    assert res.status_code == 422


def test_create_training_run_refuses_a_missing_data_dir_with_no_real_data(client, tmp_path):
    res = client.post(
        "/api/training/runs",
        json={"base_model": "test-model", "data_dir": "nothing_here", "objective": "grpo"},
    )

    assert res.status_code == 422
    assert "no training data" in res.json()["detail"].lower()
    assert not (tmp_path / "nothing_here").exists()


def test_create_training_run_auto_creates_missing_data_dir(client, tmp_path):
    _seed_rollouts({"alpha": 1})
    with patch("backend.app.api.training.threading.Thread"):
        res = client.post(
            "/api/training/runs",
            json={
                "base_model": "test-model",
                "data_dir": "auto_created_dir_12345",
                "objective": "grpo",
            },
        )
        assert res.status_code == 202
        assert (tmp_path / "auto_created_dir_12345").exists()


def test_get_training_data_sources(client):
    res = client.get("/api/training/data-sources")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert any(s["id"] == "exports" for s in data)


def _seed_rollouts(episodes_per_env: dict[str, int]) -> None:
    from datetime import datetime, timedelta, timezone
    from backend.app.models import Episode, SandboxEnvironment

    with get_session_factory()() as db:
        for env_name, count in episodes_per_env.items():
            db.add(SandboxEnvironment(
                id=env_name, status="running", env_type="general",
                expires_at=datetime.now(timezone.utc) + timedelta(days=1),
            ))
            for i in range(count):
                db.add(Episode(
                    id=f"{env_name}_{i}", env_name=env_name, task_name="t", seed=i,
                    agent_id="random", started_at=datetime.now(timezone.utc),
                ))
        db.commit()


def test_data_sources_list_each_environment_with_rollouts_and_its_count(client):
    _seed_rollouts({"alpha": 2, "beta": 1, "empty": 0})

    data = client.get("/api/training/data-sources").json()

    env_sources = {s["env_name"]: s["label"] for s in data if s["source_type"] == "environment"}
    assert env_sources == {
        "alpha": "Environment Rollouts: alpha (2 episodes)",
        "beta": "Environment Rollouts: beta (1 episodes)",
    }


def test_data_sources_count_episodes_in_one_query_regardless_of_env_count(client):
    from sqlalchemy import event
    from backend.app.database import get_engine

    _seed_rollouts({f"env{i}": 1 for i in range(5)})
    episode_queries: list[str] = []

    def record(_conn, _cursor, statement, *_args):
        if "FROM episodes" in statement:
            episode_queries.append(statement)

    event.listen(get_engine(), "before_cursor_execute", record)
    try:
        client.get("/api/training/data-sources")
    finally:
        event.remove(get_engine(), "before_cursor_execute", record)

    assert len(episode_queries) == 1


def test_create_and_get_training_run(client, tmp_path):
    data_dir = tmp_path / "train_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out_policy"

    with patch("backend.app.api.training.threading.Thread") as mock_thread:
        mock_instance = mock_thread.return_value
        res = client.post(
            "/api/training/runs",
            json={
                "base_model": "meta-llama/Llama-3-8B-Instruct",
                "training_mode": "offline",
                "data_dir": "train_data",
                "output_dir": "out_policy",
                "objective": "grpo",
                "max_steps": 100,
            },
        )
        assert res.status_code == 202
        run_id = res.json()["run_id"]
        assert run_id.startswith("tr_")
        assert mock_instance.start.called

    # List runs
    list_res = client.get("/api/training/runs")
    assert list_res.status_code == 200
    runs = list_res.json()
    assert len(runs) >= 1
    assert runs[0]["id"] == run_id
    assert runs[0]["objective"] == "grpo"
    assert runs[0]["training_mode"] == "offline"

    # Get run details
    get_res = client.get(f"/api/training/runs/{run_id}")
    assert get_res.status_code == 200
    assert get_res.json()["base_model"] == "meta-llama/Llama-3-8B-Instruct"
    assert get_res.json()["training_mode"] == "offline"


def test_list_checkpoints(client, tmp_path):
    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a dummy checkpoint
    cp = PolicyCheckpoint(
        objective="grpo",
        training_mode="offline",
        base_model="test-base",
        model_path="test-base",
        num_examples=100,
        mean_reward=0.85,
        run_id="tr_sample",
    )
    cp.save(out_dir / "cp_1")

    res = client.get("/api/training/checkpoints?output_dir=checkpoints")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 1
    assert data[0]["objective"] == "grpo"
    assert data[0]["training_mode"] == "offline"
    assert data[0]["num_examples"] == 100
