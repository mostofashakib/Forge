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


def _run_with_report(run_id: str, report) -> None:
    with get_session_factory()() as db:
        db.add(BenchmarkRun(
            id=run_id, status="done", domains="cli", depth=3, seeds=2,
            output_dir="benchmark_results", created_at=datetime.now(timezone.utc),
            report_json=json.dumps(report),
        ))
        db.commit()


def test_env_quality_runs_have_no_pass_samples_to_chart(client):
    _run_with_report("bm_quality", [{
        "env_name": "cli_env_1", "state_coverage_score": 0.8, "reward_density": 0.75,
        "dead_end_rate": 0.1, "action_diversity": 0.85, "num_episodes": 10, "num_steps": 100,
    }])

    res = client.get("/api/benchmark/runs/bm_quality/graphs?max_k=5")

    assert res.status_code == 422
    assert "no per-task pass/fail samples" in res.json()["detail"]


def test_held_out_eval_runs_chart_their_measured_task_samples(client):
    _run_with_report("eval_1", {
        "heldout_pass_rate": 0.5,
        "task_pass_counts": {
            "held_a/t1": {"decided": 4, "passed": 4},
            "held_a/t2": {"decided": 4, "passed": 0},
            # Every sample abstained: nothing measured, nothing charted.
            "held_b/t3": {"decided": 0, "passed": 0},
        },
    })

    data = client.get("/api/benchmark/runs/eval_1/graphs?max_k=2").json()

    assert data["pass_at_k"] == [0.5, 0.5]
    assert data["pass_pow_k"] == [0.5, 0.5]


def test_a_harbor_exit_status_is_not_charted_as_a_task_pass(client):
    _run_with_report("eval_harbor", {"engine": "harbor", "status": "completed", "task_path": "tasks/x"})

    res = client.get("/api/benchmark/runs/eval_harbor/graphs")

    assert res.status_code == 422


def test_recorded_trials_are_charted_as_given(client):
    _run_with_report("bm_trials", {"trials": [{"task_id": "t1", "passed_samples": 1, "total_samples": 2}]})

    data = client.get("/api/benchmark/runs/bm_trials/graphs?max_k=1").json()

    assert data["pass_at_k"] == [0.5]


def test_posting_no_trials_is_refused_instead_of_reporting_low_risk(client):
    res = client.post("/api/benchmark/graphs", json={"trials": []})

    assert res.status_code == 422


def test_get_benchmark_run_graphs_not_found(client):
    res = client.get("/api/benchmark/runs/nonexistent/graphs")
    assert res.status_code == 404
