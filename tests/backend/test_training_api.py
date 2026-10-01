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


def test_create_training_run_missing_data_dir(client):
    res = client.post(
        "/api/training/runs",
        json={
            "base_model": "test-model",
            "data_dir": "nonexistent_dir_12345",
            "objective": "grpo",
        },
    )
    assert res.status_code == 422


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

    # Get run details
    get_res = client.get(f"/api/training/runs/{run_id}")
    assert get_res.status_code == 200
    assert get_res.json()["base_model"] == "meta-llama/Llama-3-8B-Instruct"


def test_list_checkpoints(client, tmp_path):
    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a dummy checkpoint
    cp = PolicyCheckpoint(
        objective="grpo",
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
    assert data[0]["num_examples"] == 100
