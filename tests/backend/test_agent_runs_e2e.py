"""
End-to-end tests for the full todo-app general-environment creation and agent run flow.

These tests cover:
  1. Creating a general (HTTP) environment from a todo-app description
  2. The Celery build task writing a Dockerfile, pulling the base image (with
     the new per-attempt timeout), docker-building, and publishing progress
  3. Launching an agent run on a running sandbox
  4. Validation guards (non-running sandbox, missing container_id, etc.)
  5. Listing, fetching, and exporting agent runs / episodes
  6. The trajectory endpoint serving JSONL step records
  7. Deleting a run (cascades to episodes + trajectory files)
  8. Cross-run selection state — each agent run tracks its own selected episodes
     independently (regression test for the onSelectionChange wiring)

All external dependencies (Redis, Celery, Docker) are mocked so tests run
without any running services.
"""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import docker.errors
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import SandboxEnvironment, AgentRun, AgentEpisode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient backed by an isolated in-memory SQLite DB."""
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from backend.app.main import app
    return TestClient(app)


def _add_running_general_sandbox(client, env_name: str, port: int = 9001) -> None:
    """Insert a fully-running general sandbox directly into the DB."""
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        db.add(SandboxEnvironment(
            id=env_name,
            status="running",
            env_type="general",
            container_id="fake-container-abc",
            container_port=port,
            image_tag=f"forge-env-{env_name}:latest",
            ttl_days=7,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        ))
        db.commit()
    finally:
        db.close()


def _add_running_cli_sandbox(client, env_name: str) -> None:
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        db.add(SandboxEnvironment(
            id=env_name,
            status="running",
            env_type="cli",
            container_id="fake-cli-container",
            container_port=None,
            image_tag="builtin:cli",
            ttl_days=7,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        ))
        db.commit()
    finally:
        db.close()


def _mock_sandbox_create():
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    return [
        patch("backend.app.api.sandbox.redis.from_url", return_value=mock_redis),
        patch("backend.app.worker.tasks.build_sandbox_task.delay",
              return_value=MagicMock(id="mock-task-id")),
    ]


# ---------------------------------------------------------------------------
# 1. Todo-app general environment creation flow
# ---------------------------------------------------------------------------

def test_todo_app_creation_queues_successfully(client):
    """POST /api/sandbox/ for a todo app sets status=queued and returns job_id."""
    mocks = _mock_sandbox_create()
    with mocks[0], mocks[1]:
        resp = client.post("/api/sandbox/", json={
            "env_name": "todo_clone",
            "description": "A simple todo list app",
            "domain": "productivity",
            "ttl_days": 7,
        })
    assert resp.status_code == 202
    body = resp.json()
    assert body["env_name"] == "todo_clone"
    assert "job_id" in body

    info = client.get("/api/sandbox/todo_clone").json()
    assert info["status"] == "queued"
    assert info["env_type"] == "general"


def test_todo_app_dockerfile_pull_uses_per_attempt_timeout(tmp_path):
    """
    ContainerRuntime.build() with a python:3.11-slim Dockerfile (LLM-generated style)
    must:
      1. Normalise the FROM line to FORGE_PYTHON_BASE (canonical Forge base).
      2. Pass timeout=120 to the docker pull subprocess call so a hung Hub
         connection never blocks the worker indefinitely.
    Together with worker-startup pre-warming, the canonical base is normally
    cached and the pull is skipped entirely — but when it does run, it has
    a hard wall-clock cap.
    """
    from forge.envgen.container import ContainerRuntime, FORGE_PYTHON_BASE

    app_dir = tmp_path / "todo_build"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\n"
        "RUN pip install fastapi uvicorn\n"
        'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
    )
    (app_dir / "main.py").write_text('from fastapi import FastAPI\napp = FastAPI()\n')

    with patch("forge.envgen.container.subprocess.run") as mock_subproc, \
         patch("forge.envgen.container._image_cached_locally", return_value=False):
        mock_subproc.return_value = MagicMock(returncode=0)
        ContainerRuntime().build("todo_build", app_dir)

    pull_calls = [c for c in mock_subproc.call_args_list if c.args[0][1] == "pull"]
    assert len(pull_calls) == 1
    # Pulled the canonical base, NOT the LLM-chosen 3.11-slim
    assert pull_calls[0].args[0][2] == FORGE_PYTHON_BASE
    assert pull_calls[0].kwargs["timeout"] == 120


def test_todo_app_build_skips_pull_when_image_already_cached(tmp_path):
    """If python:3.11-slim is already in the local cache, no pull is attempted."""
    app_dir = tmp_path / "todo_cached"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /app\n")
    (app_dir / "main.py").write_text("")

    with patch("forge.envgen.container.subprocess.run") as mock_subproc, \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        mock_subproc.return_value = MagicMock(returncode=0)
        from forge.envgen.container import ContainerRuntime
        ContainerRuntime().build("todo_cached", app_dir)

    pull_calls = [c for c in mock_subproc.call_args_list if c.args[0][1] == "pull"]
    assert len(pull_calls) == 0, "No docker pull when image is already in local cache"


def test_todo_app_build_retries_on_eof_then_succeeds(tmp_path):
    """If the first pull attempt hits EOF, the retry succeeds and docker build runs.

    The Dockerfile has been normalised to FORGE_PYTHON_BASE before any pull
    happens, so the retry targets the canonical base.
    """
    from forge.envgen.container import ContainerRuntime, FORGE_PYTHON_BASE

    app_dir = tmp_path / "todo_eof_retry"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /app\n")
    (app_dir / "main.py").write_text("")

    eof_exc = subprocess.CalledProcessError(
        1, "docker pull",
        stderr="failed to do request: Head https://registry-1.docker.io/...: EOF",
    )

    with patch("forge.envgen.container.subprocess.run") as mock_subproc, \
         patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container.time.sleep"):
        mock_subproc.side_effect = [eof_exc, MagicMock(returncode=0), MagicMock(returncode=0)]
        ContainerRuntime().build("todo_eof_retry", app_dir)

    calls = [c.args[0] for c in mock_subproc.call_args_list]
    assert calls[0] == ["docker", "pull", FORGE_PYTHON_BASE]
    assert calls[1] == ["docker", "pull", FORGE_PYTHON_BASE]
    assert calls[2][1] == "build"


def test_todo_app_build_fails_when_all_pull_retries_exhausted(tmp_path):
    """If all pull attempts fail, build() raises RuntimeError and docker build is not run."""
    from forge.envgen.container import ContainerRuntime, FORGE_PYTHON_BASE

    app_dir = tmp_path / "todo_pull_fail"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /app\n")
    (app_dir / "main.py").write_text("")

    eof_exc = subprocess.CalledProcessError(1, "docker pull", stderr="EOF")

    with patch("forge.envgen.container.subprocess.run", side_effect=eof_exc), \
         patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container.time.sleep"), \
         patch("forge.envgen._image_pull_http.pull_via_http",
               side_effect=RuntimeError("HTTPS also EOF")):
        with pytest.raises(RuntimeError, match=f"Failed to pull {FORGE_PYTHON_BASE}"):
            ContainerRuntime().build("todo_pull_fail", app_dir)


# ---------------------------------------------------------------------------
# 2. Agent run creation and validation
# ---------------------------------------------------------------------------

def test_create_agent_run_on_running_sandbox(client):
    """POST agent-runs on a running general sandbox returns 202 with run_id."""
    _add_running_general_sandbox(client, "todo_agent_env")

    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        resp = client.post(
            "/api/sandbox/todo_agent_env/agent-runs",
            json={"objective": "Add three todo items and mark one complete", "num_episodes": 3},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert "run_id" in body
    assert body["objective"] == "Add three todo items and mark one complete"
    assert body["status"] == "pending"
    assert body["num_episodes"] == 3


def test_create_agent_run_on_cli_sandbox(client):
    """CLI sandboxes (no port) are also accepted for agent runs."""
    _add_running_cli_sandbox(client, "cli_agent_env")

    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        resp = client.post(
            "/api/sandbox/cli_agent_env/agent-runs",
            json={"objective": "List all files in /home", "num_episodes": 2},
        )
    assert resp.status_code == 202


def test_create_agent_run_rejects_non_running_sandbox(client):
    """Agent run creation must fail 409 if sandbox is not running."""
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        db.add(SandboxEnvironment(
            id="stopped_env",
            status="stopped",
            env_type="general",
            ttl_days=7,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        ))
        db.commit()
    finally:
        db.close()

    resp = client.post(
        "/api/sandbox/stopped_env/agent-runs",
        json={"objective": "Do something"},
    )
    assert resp.status_code == 409
    assert "running" in resp.json()["detail"].lower()


def test_create_agent_run_rejects_sandbox_without_container(client):
    """A running sandbox with no container_id must return 409."""
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        db.add(SandboxEnvironment(
            id="nocontainer_env",
            status="running",
            env_type="general",
            container_id=None,
            ttl_days=7,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        ))
        db.commit()
    finally:
        db.close()

    resp = client.post(
        "/api/sandbox/nocontainer_env/agent-runs",
        json={"objective": "Do something"},
    )
    assert resp.status_code == 409


def test_create_agent_run_rejects_general_sandbox_without_port(client):
    """General sandbox with no container_port must return 409."""
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        db.add(SandboxEnvironment(
            id="noport_env",
            status="running",
            env_type="general",
            container_id="some-id",
            container_port=None,
            ttl_days=7,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        ))
        db.commit()
    finally:
        db.close()

    resp = client.post(
        "/api/sandbox/noport_env/agent-runs",
        json={"objective": "Do something"},
    )
    assert resp.status_code == 409


def test_create_agent_run_returns_404_for_unknown_env(client):
    resp = client.post(
        "/api/sandbox/ghost_env/agent-runs",
        json={"objective": "Do something"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 3. Listing agent runs and episodes
# ---------------------------------------------------------------------------

def test_list_agent_runs_returns_all_runs_for_env(client):
    """GET /api/sandbox/{env}/agent-runs lists only that env's runs."""
    _add_running_general_sandbox(client, "multi_run_env")

    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        for obj in ["objective A", "objective B", "objective C"]:
            client.post(
                "/api/sandbox/multi_run_env/agent-runs",
                json={"objective": obj, "num_episodes": 1},
            )

    runs = client.get("/api/sandbox/multi_run_env/agent-runs").json()
    assert len(runs) == 3
    objectives = {r["objective"] for r in runs}
    assert objectives == {"objective A", "objective B", "objective C"}


def test_list_episodes_returns_empty_initially(client):
    """Episodes list is empty before any Celery worker processes the run."""
    _add_running_general_sandbox(client, "fresh_run_env")

    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        run_id = client.post(
            "/api/sandbox/fresh_run_env/agent-runs",
            json={"objective": "initial", "num_episodes": 5},
        ).json()["run_id"]

    eps = client.get(f"/api/sandbox/fresh_run_env/agent-runs/{run_id}/episodes").json()
    assert eps == []


def test_list_episodes_after_worker_populates_db(client):
    """After the worker writes episodes to DB, GET /episodes returns them."""
    _add_running_general_sandbox(client, "populated_env")

    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        run_id = client.post(
            "/api/sandbox/populated_env/agent-runs",
            json={"objective": "check list", "num_episodes": 3},
        ).json()["run_id"]

    # Simulate worker writing 3 completed episodes
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        for idx in range(3):
            db.add(AgentEpisode(
                id=str(uuid.uuid4()),
                run_id=run_id,
                episode_index=idx,
                seed=idx,
                status="completed",
                total_steps=10 + idx,
                total_reward=0.4 + idx * 0.1,
                final_objective_score=0.5 + idx * 0.1,
                termination_reason="success" if idx == 2 else "max_steps",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            ))
        run = db.get(AgentRun, run_id)
        run.status = "completed"
        run.episodes_completed = 3
        run.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()

    eps = client.get(f"/api/sandbox/populated_env/agent-runs/{run_id}/episodes").json()
    assert len(eps) == 3
    assert eps[0]["episode_index"] == 0
    assert eps[2]["termination_reason"] == "success"
    # Rewards are normalized to [0, 1]
    for ep in eps:
        assert 0.0 <= ep["total_reward"] <= 1.0


# ---------------------------------------------------------------------------
# 4. Trajectory endpoint
# ---------------------------------------------------------------------------

def test_get_trajectory_serves_jsonl_steps(client, tmp_path):
    """GET .../trajectory returns steps parsed from the JSONL file on disk."""
    _add_running_general_sandbox(client, "traj_env")

    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        run_id = client.post(
            "/api/sandbox/traj_env/agent-runs",
            json={"objective": "traj test", "num_episodes": 1},
        ).json()["run_id"]

    # Write a JSONL trajectory file and insert episode record pointing to it
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    traj_file = traj_dir / "ep0.jsonl"
    steps = [
        {"step_index": 0, "action": {"endpoint": "/todos", "payload": {}},
         "state_before": {}, "state_after": {"todos": [{"id": 1}]},
         "reward": 0.5, "objective_score": 0.5, "state_hash_before": "a",
         "state_hash_after": "b", "terminated": False, "truncated": False,
         "termination_reason": None},
        {"step_index": 1, "action": {"endpoint": "/todos/1/complete", "payload": {}},
         "state_before": {}, "state_after": {},
         "reward": 1.0, "objective_score": 1.0, "state_hash_before": "b",
         "state_hash_after": "c", "terminated": True, "truncated": False,
         "termination_reason": "success"},
    ]
    summary = {
        "type": "episode_summary",
        "total_steps": 2,
        "total_reward": 0.75,
        "final_objective_score": 1.0,
        "termination_reason": "success",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    traj_file.write_text("\n".join(json.dumps(r) for r in [*steps, summary]))

    ep_id = str(uuid.uuid4())
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        db.add(AgentEpisode(
            id=ep_id,
            run_id=run_id,
            episode_index=0,
            seed=0,
            status="completed",
            total_steps=2,
            total_reward=0.75,
            final_objective_score=1.0,
            termination_reason="success",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            jsonl_path=str(traj_file),
        ))
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/api/sandbox/traj_env/agent-runs/{run_id}/episodes/{ep_id}/trajectory")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["steps"]) == 2
    assert body["steps"][0]["step_index"] == 0
    assert body["steps"][1]["termination_reason"] == "success"
    assert body["summary"]["termination_reason"] == "success"
    assert body["summary"]["total_reward"] == 0.75


def test_trajectory_returns_404_when_file_missing(client):
    """If the JSONL file was deleted or never written, GET /trajectory → 404."""
    _add_running_general_sandbox(client, "notraj_env")

    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        run_id = client.post(
            "/api/sandbox/notraj_env/agent-runs",
            json={"objective": "no traj", "num_episodes": 1},
        ).json()["run_id"]

    ep_id = str(uuid.uuid4())
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        db.add(AgentEpisode(
            id=ep_id, run_id=run_id, episode_index=0, seed=0,
            status="completed", total_steps=0, total_reward=0.0,
            final_objective_score=0.0, started_at=datetime.now(timezone.utc),
            jsonl_path="/nonexistent/path.jsonl",
        ))
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/api/sandbox/notraj_env/agent-runs/{run_id}/episodes/{ep_id}/trajectory")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. Export endpoint
# ---------------------------------------------------------------------------

def test_export_trajectories_concatenates_jsonl_files(client, tmp_path):
    """GET /export returns all completed episode JSONL files concatenated."""
    _add_running_general_sandbox(client, "export_env")

    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        run_id = client.post(
            "/api/sandbox/export_env/agent-runs",
            json={"objective": "export test", "num_episodes": 2},
        ).json()["run_id"]

    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()

    ep_ids = []
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        for idx in range(2):
            ep_id = str(uuid.uuid4())
            ep_ids.append(ep_id)
            f = traj_dir / f"ep{idx}.jsonl"
            f.write_text(json.dumps({"step_index": 0, "reward": 0.5 + idx * 0.3}))
            db.add(AgentEpisode(
                id=ep_id, run_id=run_id, episode_index=idx, seed=idx,
                status="completed", total_steps=1,
                total_reward=0.5 + idx * 0.3, final_objective_score=0.8,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                jsonl_path=str(f),
            ))
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/api/sandbox/export_env/agent-runs/{run_id}/export")
    assert resp.status_code == 200
    assert "ndjson" in resp.headers["content-type"]
    lines = [l for l in resp.text.strip().splitlines() if l]
    assert len(lines) == 2
    records = [json.loads(l) for l in lines]
    rewards = {r["reward"] for r in records}
    assert 0.5 in rewards
    assert 0.8 in rewards


# ---------------------------------------------------------------------------
# 6. Delete run (cascade)
# ---------------------------------------------------------------------------

def test_delete_run_removes_db_rows_and_trajectory_files(client, tmp_path):
    """DELETE /agent-runs/{run_id} removes the run, episodes, and JSONL files."""
    _add_running_general_sandbox(client, "delete_env")

    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        run_id = client.post(
            "/api/sandbox/delete_env/agent-runs",
            json={"objective": "to delete", "num_episodes": 1},
        ).json()["run_id"]

    traj_file = tmp_path / "ep0.jsonl"
    traj_file.write_text(json.dumps({"step_index": 0, "reward": 0.3}))

    ep_id = str(uuid.uuid4())
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        db.add(AgentEpisode(
            id=ep_id, run_id=run_id, episode_index=0, seed=0,
            status="completed", total_steps=1, total_reward=0.3,
            final_objective_score=0.3, started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            jsonl_path=str(traj_file),
        ))
        db.commit()
    finally:
        db.close()

    assert traj_file.exists()
    resp = client.delete(f"/api/sandbox/delete_env/agent-runs/{run_id}")
    assert resp.status_code == 204

    # JSONL file must be gone
    assert not traj_file.exists()

    # Episodes must be gone
    eps = client.get(f"/api/sandbox/delete_env/agent-runs/{run_id}/episodes")
    assert eps.status_code == 404


def test_delete_nonexistent_run_returns_404(client):
    _add_running_general_sandbox(client, "del404_env")
    resp = client.delete("/api/sandbox/del404_env/agent-runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 7. Cross-run episode selection regression
#    (Verifies the API layer that feeds the onSelectionChange UI)
# ---------------------------------------------------------------------------

def test_multiple_runs_have_independent_episode_lists(client):
    """
    Each agent run has an independent episode list.
    Selecting episodes from run A must not affect run B.
    This tests the API contract that the onSelectionChange UI relies on.
    """
    _add_running_general_sandbox(client, "selection_env")

    run_ids = []
    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        for obj in ["run A objective", "run B objective"]:
            run_id = client.post(
                "/api/sandbox/selection_env/agent-runs",
                json={"objective": obj, "num_episodes": 2},
            ).json()["run_id"]
            run_ids.append(run_id)

    # Populate episodes for both runs
    from backend.app import database
    db: Session = database.get_session_factory()()
    ep_ids_by_run: dict[str, list[str]] = {}
    try:
        for run_id in run_ids:
            ep_ids_by_run[run_id] = []
            for idx in range(2):
                ep_id = str(uuid.uuid4())
                ep_ids_by_run[run_id].append(ep_id)
                db.add(AgentEpisode(
                    id=ep_id, run_id=run_id, episode_index=idx, seed=idx,
                    status="completed", total_steps=5, total_reward=0.6,
                    final_objective_score=0.6, started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                ))
        db.commit()
    finally:
        db.close()

    run_a_id, run_b_id = run_ids

    # Episodes for run A and run B are distinct sets
    eps_a = client.get(f"/api/sandbox/selection_env/agent-runs/{run_a_id}/episodes").json()
    eps_b = client.get(f"/api/sandbox/selection_env/agent-runs/{run_b_id}/episodes").json()

    assert len(eps_a) == 2
    assert len(eps_b) == 2
    ids_a = {e["id"] for e in eps_a}
    ids_b = {e["id"] for e in eps_b}
    assert ids_a.isdisjoint(ids_b), "Run A and run B must have completely different episode IDs"

    # All episodes in run A belong only to run A
    for ep in eps_a:
        assert ep["run_id"] == run_a_id
    for ep in eps_b:
        assert ep["run_id"] == run_b_id


def test_cross_run_episode_export_merges_trajectories(client, tmp_path):
    """
    Exporting selected episodes from two different runs produces a merged JSONL.
    This mirrors what the DataCollectionPanel does on the frontend.
    """
    _add_running_general_sandbox(client, "merge_env")

    run_ids = []
    with patch("backend.app.worker.tasks.run_container_run_task.delay"):
        for _ in range(2):
            run_id = client.post(
                "/api/sandbox/merge_env/agent-runs",
                json={"objective": "merge test", "num_episodes": 1},
            ).json()["run_id"]
            run_ids.append(run_id)

    traj_dir = tmp_path / "merge_trajs"
    traj_dir.mkdir()

    from backend.app import database
    db: Session = database.get_session_factory()()
    ep_ids = []
    try:
        for i, run_id in enumerate(run_ids):
            ep_id = str(uuid.uuid4())
            ep_ids.append(ep_id)
            f = traj_dir / f"ep_{i}.jsonl"
            f.write_text(json.dumps({"step_index": 0, "run_tag": f"run{i}", "reward": 0.7}))
            db.add(AgentEpisode(
                id=ep_id, run_id=run_id, episode_index=0, seed=i,
                status="completed", total_steps=1, total_reward=0.7,
                final_objective_score=0.7, started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                jsonl_path=str(f),
            ))
        db.commit()
    finally:
        db.close()

    # Fetch trajectories from both runs (simulating what DataCollectionPanel does)
    all_steps = []
    for run_id, ep_id in zip(run_ids, ep_ids):
        resp = client.get(
            f"/api/sandbox/merge_env/agent-runs/{run_id}/episodes/{ep_id}/trajectory"
        )
        assert resp.status_code == 200
        all_steps.extend(resp.json()["steps"])

    assert len(all_steps) == 2
    run_tags = {s["run_tag"] for s in all_steps}
    assert run_tags == {"run0", "run1"}
