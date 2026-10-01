"""Tests for benchmark API graph routes (Feature Request 9)."""
import json
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import BenchmarkRun
from backend.app.database import get_session_factory, init_db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test_bm_graphs.db")
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    init_db()
    return TestClient(app)


def test_post_custom_benchmark_graphs(client):
    trials_payload = [
        {
            "task_id": "math_1",
            "passed_samples": 4,
            "total_samples": 5,
            "paraphrased_passed_samples": 4,
            "paraphrased_total_samples": 5,
            "verifier_passes": [True, True, True, True, False],
            "ground_truth_passes": [True, True, True, True, False],
        },
        {
            "task_id": "math_2",
            "passed_samples": 3,
            "total_samples": 5,
            "paraphrased_passed_samples": 3,
            "paraphrased_total_samples": 5,
            "verifier_passes": [True, True, True, False, False],
            "ground_truth_passes": [True, True, True, False, False],
        },
    ]

    res = client.post(
        "/api/benchmark/graphs",
        json={"trials": trials_payload, "max_k": 5, "graph_type": "all"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "k_labels" in data
    assert len(data["pass_at_k"]) == 5
    assert len(data["pass_pow_k"]) == 5
    assert data["cohens_kappa"] == 1.0
    assert "pass_curve" in data["chart_configs"]
    assert "agreement_bar" in data["chart_configs"]
    assert "risk_breakdown" in data["chart_configs"]
    assert data["diagnostics"]["genuine_learning_index"] > 0


def test_post_filtered_graph_type(client):
    trials_payload = [
        {"task_id": "t1", "passed_samples": 5, "total_samples": 5}
    ]
    res = client.post(
        "/api/benchmark/graphs",
        json={"trials": trials_payload, "graph_type": "pass_curve"},
    )
    assert res.status_code == 200
    data = res.json()
    assert list(data["chart_configs"].keys()) == ["pass_curve"]


def test_get_benchmark_run_graphs(client):
    SessionLocal = get_session_factory()
    with SessionLocal() as db:
        run = BenchmarkRun(
            id="bm_graph_run_1",
            status="done",
            domains="cli",
            depth=3,
            seeds=2,
            output_dir="benchmark_results",
            created_at=datetime.now(timezone.utc),
            report_json=json.dumps([
                {
                    "env_name": "cli_env_1",
                    "state_coverage_score": 0.8,
                    "reward_density": 0.75,
                    "dead_end_rate": 0.1,
                    "action_diversity": 0.85,
                    "num_episodes": 10,
                    "num_steps": 100,
                }
            ]),
        )
        db.add(run)
        db.commit()

    res = client.get("/api/benchmark/runs/bm_graph_run_1/graphs?max_k=5")
    assert res.status_code == 200
    data = res.json()
    assert len(data["pass_at_k"]) == 5
    assert "chart_configs" in data
    assert "pass_curve" in data["chart_configs"]


def test_get_benchmark_run_graphs_not_found(client):
    res = client.get("/api/benchmark/runs/nonexistent/graphs")
    assert res.status_code == 404
