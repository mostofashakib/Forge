"""
End-to-end tests for the sandbox environment creation flow.

These tests cover the full HTTP → DB → Celery dispatch path using
in-memory SQLite and mocked external dependencies (Redis, Celery, Docker).
"""
import asyncio
import logging

import docker.errors
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import SandboxEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient backed by an isolated in-memory DB and a mocked tmp envs dir."""
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from backend.app.main import app
    return TestClient(app)


def _add_sandbox(client, env_name: str, status: str = "running") -> None:
    """Directly insert a SandboxEnvironment row (bypasses the POST endpoint)."""
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        db.add(SandboxEnvironment(
            id=env_name,
            status=status,
            ttl_days=30,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        ))
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helper to mock the two external calls in POST /api/sandbox/
# ---------------------------------------------------------------------------

def _mock_create_deps():
    """
    Context manager stack that patches away the Redis health check and
    Celery task dispatch so POST /api/sandbox/ completes synchronously.

    Both .ping() and .delay() run inside run_in_executor threads, so
    regular MagicMocks work fine.
    """
    mock_redis_inst = MagicMock()
    mock_redis_inst.ping.return_value = True

    return [
        # redis.from_url(...).ping() → True
        patch(
            "backend.app.api.sandbox.redis.from_url",
            return_value=mock_redis_inst,
        ),
        # Celery .delay() returns a mock AsyncResult
        patch(
            "backend.app.worker.tasks.build_sandbox_task.delay",
            return_value=MagicMock(id="mock-task-id"),
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_sandbox_queues_successfully(client):
    """POST /api/sandbox/ creates a DB row with status=queued and returns job_id."""
    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        resp = client.post("/api/sandbox/", json={
            "env_name": "ticket_env",
            "description": "A support ticket system",
            "domain": "support",
            "ttl_days": 7,
        })
    assert resp.status_code == 202
    body = resp.json()
    assert body["env_name"] == "ticket_env"
    assert "job_id" in body

    # DB row should now exist with status=queued
    resp2 = client.get("/api/sandbox/ticket_env")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "queued"


def test_create_sandbox_redis_failure_leaves_no_record(client):
    redis_client = MagicMock()
    redis_client.ping.side_effect = ConnectionError("redis unavailable")

    with patch("backend.app.api.sandbox.redis.from_url", return_value=redis_client):
        response = client.post("/api/sandbox/", json={"env_name": "redis_failure"})

    assert response.status_code == 503
    assert client.get("/api/sandbox/redis_failure").status_code == 404


def test_create_sandbox_dispatch_failure_removes_queued_record(client):
    redis_patch, task_patch = _mock_create_deps()
    with redis_patch, task_patch as task_delay:
        task_delay.side_effect = RuntimeError("broker rejected message")
        response = client.post("/api/sandbox/", json={"env_name": "dispatch_failure"})

    assert response.status_code == 503
    assert client.get("/api/sandbox/dispatch_failure").status_code == 404


def test_capacity_uses_backend_configuration(client, monkeypatch):
    monkeypatch.setenv("FORGE_SANDBOX_LIMIT", "17")
    _add_sandbox(client, "active_env", status="running")
    _add_sandbox(client, "deleted_env", status="deleted")

    response = client.get("/api/sandbox/capacity")

    assert response.status_code == 200
    assert response.json() == {"active_count": 1, "limit": 17}


def test_create_sandbox_requires_product_url_for_user_research(client):
    response = client.post("/api/sandbox/", json={
        "env_name": "researched_env",
        "description": "A project tracker",
        "use_user_researcher": True,
    })

    assert response.status_code == 422
    assert "source_product_url must be a valid" in response.text


def test_create_sandbox_infers_product_name_from_research_url(client):
    mocks = _mock_create_deps()
    with mocks[0], mocks[1] as task_delay:
        response = client.post("/api/sandbox/", json={
            "env_name": "researched_env",
            "use_user_researcher": True,
            "source_product_url": "https://github.com/acme/project-board",
        })

    assert response.status_code == 202
    assert task_delay.call_args.kwargs["source_product_name"] == "Project Board"


def test_create_sandbox_forwards_user_research_selection(client):
    mocks = _mock_create_deps()
    with mocks[0], mocks[1] as task_delay:
        response = client.post("/api/sandbox/", json={
            "env_name": "researched_env",
            "description": "A project tracker",
            "use_user_researcher": True,
            "source_product_name": "Linear",
            "source_product_url": "https://linear.app",
            "reference_urls": ["https://linear.app/docs"],
        })

    assert response.status_code == 202
    task_kwargs = task_delay.call_args.kwargs
    assert task_kwargs["use_user_researcher"] is True
    assert task_kwargs["source_product_name"] == "Linear"
    assert task_kwargs["source_product_url"] == "https://linear.app"


def test_create_sandbox_duplicate_name_rejected(client):
    """Creating a sandbox with a name that already exists returns 409."""
    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        r1 = client.post("/api/sandbox/", json={
            "env_name": "dup_env",
            "description": "first",
            "ttl_days": 7,
        })
    assert r1.status_code == 202

    mocks2 = _mock_create_deps()
    with mocks2[0], mocks2[1]:
        r2 = client.post("/api/sandbox/", json={
            "env_name": "dup_env",
            "description": "second attempt",
            "ttl_days": 7,
        })
    assert r2.status_code == 409
    assert "already exists" in r2.json()["detail"]


def test_create_sandbox_reuses_deleted_name(client):
    """A deleted/expired sandbox row allows re-creation with the same name."""
    _add_sandbox(client, "reusable_env", status="deleted")

    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        resp = client.post("/api/sandbox/", json={
            "env_name": "reusable_env",
            "description": "fresh start",
            "ttl_days": 14,
        })
    assert resp.status_code == 202


def test_create_sandbox_enforces_10_env_limit(client):
    """POST /api/sandbox/ returns 429 when 10 active sandboxes already exist."""
    for i in range(10):
        _add_sandbox(client, f"env_{i:02d}", status="running")

    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        resp = client.post("/api/sandbox/", json={
            "env_name": "env_overflow",
            "description": "should be rejected",
            "ttl_days": 7,
        })
    assert resp.status_code == 429
    assert "limit" in resp.json()["detail"].lower()


def test_create_sandbox_limit_excludes_deleted(client):
    """Deleted/expired envs do not count toward the 10-env limit."""
    for i in range(10):
        _add_sandbox(client, f"dead_{i:02d}", status="deleted")

    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        resp = client.post("/api/sandbox/", json={
            "env_name": "new_env",
            "description": "should be allowed",
            "ttl_days": 7,
        })
    assert resp.status_code == 202


def test_list_sandboxes_excludes_deleted(client):
    """GET /api/sandbox/ only returns active (non-deleted/expired) sandboxes."""
    _add_sandbox(client, "active_env", status="running")
    _add_sandbox(client, "dead_env", status="deleted")
    _add_sandbox(client, "old_env", status="expired")

    resp = client.get("/api/sandbox/")
    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()]
    assert "active_env" in ids
    assert "dead_env" not in ids
    assert "old_env" not in ids


def test_delete_sandbox_removes_record(client, tmp_path):
    """DELETE /api/sandbox/{name} removes the DB row (no container to stop)."""
    _add_sandbox(client, "doomed_env", status="stopped")

    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.remove = MagicMock()
        resp = client.delete("/api/sandbox/doomed_env")
    assert resp.status_code == 204

    resp2 = client.get("/api/sandbox/doomed_env")
    assert resp2.status_code == 404


def test_delete_nonexistent_sandbox_returns_404(client):
    resp = client.delete("/api/sandbox/ghost_env")
    assert resp.status_code == 404


def test_stop_sandbox_updates_status(client):
    """POST /api/sandbox/{name}/stop sets status=stopped."""
    _add_sandbox(client, "running_env", status="running")

    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.stop = MagicMock()
        resp = client.post("/api/sandbox/running_env/stop")
    assert resp.status_code == 204

    info = client.get("/api/sandbox/running_env").json()
    assert info["status"] == "stopped"


def test_start_sandbox_requires_existing_image(client):
    """POST /api/sandbox/{name}/start returns 409 when no image is built yet."""
    _add_sandbox(client, "no_image_env", status="stopped")
    # No image_tag set → should reject with 409
    resp = client.post("/api/sandbox/no_image_env/start")
    assert resp.status_code == 409
    assert "image" in resp.json()["detail"].lower()


def test_get_sandbox_resyncs_container_port_from_live_container(client):
    """When DB says running but container_port is null/stale, GET must refresh
    it from the live Docker container's port bindings — otherwise the proxy
    iframe shows 'Container not running' even though everything is fine."""
    _add_sandbox(client, "drifted_env", status="running")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "drifted_env")
        sb.image_tag = "forge-env-drifted:latest"
        sb.container_id = "live-container-id"
        sb.container_port = None  # the drift we want to heal
        db.commit()
    finally:
        db.close()

    mock_container = MagicMock()
    mock_container.status = "running"
    mock_container.attrs = {"RestartCount": 0, "State": {"Status": "running"}}
    mock_container.ports = {"8000/tcp": [{"HostPort": "32777"}]}
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container

    with patch("docker.from_env", return_value=mock_docker_client):
        info = client.get("/api/sandbox/drifted_env").json()

    # GET healed the row from the live container's port bindings
    assert info["status"] == "running"
    assert info["container_port"] == 32777


def test_get_sandbox_marks_crashing_container_as_error(client):
    """A container with RestartCount>0 is in a crash loop — even if its current
    state happens to be 'running', the UI must not pretend it's healthy."""
    _add_sandbox(client, "crash_loop", status="running")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "crash_loop")
        sb.image_tag = "forge-env-crash:latest"
        sb.container_id = "crash-id"
        sb.container_port = 32999
        db.commit()
    finally:
        db.close()

    mock_container = MagicMock()
    mock_container.status = "running"  # caught in a brief running window
    mock_container.attrs = {"RestartCount": 4, "State": {"Status": "running"}}
    mock_container.ports = {"8000/tcp": [{"HostPort": "32999"}]}
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container

    with patch("docker.from_env", return_value=mock_docker_client):
        info = client.get("/api/sandbox/crash_loop").json()

    assert info["status"] == "error"


def test_get_sandbox_marks_restarting_container_as_error(client):
    """If we catch the container in 'restarting' state explicitly, mark error."""
    _add_sandbox(client, "restart_state", status="running")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "restart_state")
        sb.image_tag = "forge-env-restart:latest"
        sb.container_id = "restart-id"
        db.commit()
    finally:
        db.close()

    mock_container = MagicMock()
    mock_container.status = "restarting"
    mock_container.attrs = {"RestartCount": 1, "State": {"Status": "restarting"}}
    mock_container.ports = {}
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container

    with patch("docker.from_env", return_value=mock_docker_client):
        info = client.get("/api/sandbox/restart_state").json()

    assert info["status"] == "error"


def test_get_sandbox_logs_docker_status_sync_failure(client, caplog):
    _add_sandbox(client, "docker_unavailable", status="running")
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sandbox = db.get(SandboxEnvironment, "docker_unavailable")
        sandbox.container_id = "container-id"
        db.commit()
    finally:
        db.close()

    with patch("docker.from_env", side_effect=RuntimeError("daemon unavailable")):
        response = client.get("/api/sandbox/docker_unavailable")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert "could not synchronize Docker status" in caplog.text


def test_get_sandbox_treats_missing_docker_socket_as_expected(client, caplog):
    _add_sandbox(client, "docker_not_running", status="running")
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sandbox = db.get(SandboxEnvironment, "docker_not_running")
        sandbox.container_id = "container-id"
        db.commit()
    finally:
        db.close()

    error = docker.errors.DockerException(
        "Error while fetching server API version",
        ("Connection aborted", FileNotFoundError(2, "No such file or directory")),
    )
    with (
        caplog.at_level(logging.DEBUG, logger="backend.app.api.sandbox"),
        patch("docker.from_env", side_effect=error),
    ):
        response = client.get("/api/sandbox/docker_not_running")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert "Docker is not running; returning persisted status" in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_stop_sandbox_surfaces_container_failure(client):
    _add_sandbox(client, "stop_failure", status="running")
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sandbox = db.get(SandboxEnvironment, "stop_failure")
        sandbox.container_id = "container-id"
        db.commit()
    finally:
        db.close()

    with patch("forge.envgen.container.ContainerRuntime") as runtime:
        runtime.return_value.stop.side_effect = RuntimeError("daemon unavailable")
        response = client.post("/api/sandbox/stop_failure/stop")

    assert response.status_code == 500
    assert "Failed to stop container" in response.json()["detail"]
    assert client.get("/api/sandbox/stop_failure").json()["status"] == "running"


def test_logs_endpoint_returns_container_output(client):
    """GET /api/sandbox/{env}/logs surfaces the crash output so users can
    diagnose why their LLM-generated app is failing on boot."""
    _add_sandbox(client, "logs_env", status="error")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "logs_env")
        sb.container_id = "container-with-logs"
        db.commit()
    finally:
        db.close()

    mock_container = MagicMock()
    mock_container.status = "exited"
    mock_container.logs.return_value = b"ModuleNotFoundError: No module named 'fastapi'\n"
    mock_container.attrs = {
        "RestartCount": 3,
        "State": {"Status": "exited", "ExitCode": 1, "Error": ""},
    }
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container

    with patch("docker.from_env", return_value=mock_docker_client):
        body = client.get("/api/sandbox/logs_env/logs").json()

    assert "ModuleNotFoundError" in body["logs"]
    assert body["exit_code"] == 1
    assert body["restart_count"] == 3
    assert body["status"] == "exited"


def test_logs_endpoint_returns_empty_when_no_container_yet(client):
    """Pre-build envs (no container_id) return empty logs, not 500."""
    _add_sandbox(client, "no_container", status="queued")
    body = client.get("/api/sandbox/no_container/logs").json()
    assert body == {"logs": "", "exit_code": None, "restart_count": 0}


def test_logs_endpoint_returns_410_when_container_pruned(client):
    """If the container_id is stale (Docker pruned), surface a clear 410."""
    _add_sandbox(client, "pruned_env", status="running")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "pruned_env")
        sb.container_id = "gone-id"
        db.commit()
    finally:
        db.close()

    import docker.errors
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.side_effect = docker.errors.NotFound("gone")

    with patch("docker.from_env", return_value=mock_docker_client):
        resp = client.get("/api/sandbox/pruned_env/logs")

    assert resp.status_code == 410
    assert "rebuilt" in resp.json()["detail"].lower()


def test_get_sandbox_demotes_to_stopped_when_running_but_no_port_binding(client):
    """Container is genuinely 'running' in Docker but has no 8000/tcp binding
    (e.g. daemon failed to allocate, or a stale container from before our
    binding fix). DB has container_port=None. Demote to 'stopped' so the
    UI shows a Start button — clicking it will run a fresh container with
    a real port binding."""
    _add_sandbox(client, "no_binding", status="running")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "no_binding")
        sb.image_tag = "forge-env-no-binding:latest"
        sb.container_id = "container-without-port"
        sb.container_port = None  # the broken-state we want to heal
        db.commit()
    finally:
        db.close()

    mock_container = MagicMock()
    mock_container.status = "running"
    mock_container.attrs = {"RestartCount": 0, "State": {"Status": "running"}}
    mock_container.ports = {}  # no 8000/tcp binding at all
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container

    with patch("docker.from_env", return_value=mock_docker_client):
        info = client.get("/api/sandbox/no_binding").json()

    # Demoted so the UI shows Start (not Stop)
    assert info["status"] == "stopped"


def test_start_sandbox_does_not_short_circuit_when_status_running_but_port_missing(client):
    """If DB says running but container_port is None, /start must NOT return
    early — it must actually run runtime.start() to fix the broken state."""
    _add_sandbox(client, "stuck_running", status="running")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "stuck_running")
        sb.image_tag = "forge-env-stuck:latest"
        sb.container_id = "stuck-id"
        sb.container_port = None  # broken state — no port
        db.commit()
    finally:
        db.close()

    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.start.return_value = ("fresh-id", 32555)
        resp = client.post("/api/sandbox/stuck_running/start")

    assert resp.status_code == 200
    body = resp.json()
    assert body["container_port"] == 32555
    mock_rt.return_value.start.assert_called_once()


def test_start_sandbox_short_circuits_for_healthy_running_general_env(client):
    """Healthy general env (running with valid port) should still short-circuit
    — we don't want every /start hit to recreate a running container."""
    _add_sandbox(client, "healthy", status="running")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "healthy")
        sb.image_tag = "forge-env-healthy:latest"
        sb.container_id = "healthy-id"
        sb.container_port = 32100
        db.commit()
    finally:
        db.close()

    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        resp = client.post("/api/sandbox/healthy/start")

    assert resp.status_code == 200
    assert resp.json()["container_port"] == 32100
    mock_rt.return_value.start.assert_not_called()


def test_start_sandbox_short_circuits_for_healthy_running_cli_env(client):
    """CLI envs intentionally have container_port=None — running CLI should
    short-circuit even though port is null."""
    _add_sandbox(client, "cli_healthy", status="running")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "cli_healthy")
        sb.image_tag = "builtin:cli"
        sb.container_id = "cli-id"
        sb.container_port = None  # correct for CLI
        db.commit()
    finally:
        db.close()

    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        resp = client.post("/api/sandbox/cli_healthy/start")

    assert resp.status_code == 200
    mock_rt.return_value.start.assert_not_called()


def test_get_sandbox_does_not_resync_port_for_cli_env(client):
    """CLI envs intentionally have no HTTP port — GET must not try to read
    8000/tcp on them, otherwise it'd silently overwrite anything the DB has."""
    _add_sandbox(client, "cli_env", status="running")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "cli_env")
        sb.image_tag = "builtin:cli"
        sb.container_id = "cli-id"
        sb.container_port = None  # CLI is correct as null
        db.commit()
    finally:
        db.close()

    mock_container = MagicMock()
    mock_container.status = "running"
    mock_container.ports = {}
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container

    with patch("docker.from_env", return_value=mock_docker_client):
        info = client.get("/api/sandbox/cli_env").json()

    assert info["status"] == "running"
    assert info["container_port"] is None


def test_start_sandbox_clears_stale_state_when_image_is_gone(client):
    """When runtime.start raises RuntimeError (image was pruned), the endpoint
    must clear the stale image_tag/container_id and return 409 — otherwise
    every subsequent /start hits the same dead image and returns 500."""
    _add_sandbox(client, "stale_env", status="stopped")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "stale_env")
        sb.image_tag = "forge-env-stale:latest"
        sb.container_id = "old-container-id"
        db.commit()
    finally:
        db.close()

    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.start.side_effect = RuntimeError(
            "Docker image 'forge-env-stale:latest' is not present locally — must be rebuilt."
        )
        resp = client.post("/api/sandbox/stale_env/start")

    assert resp.status_code == 409
    assert "rebuilt" in resp.json()["detail"].lower()

    # State has been cleared so subsequent /start returns the natural 409
    db = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "stale_env")
        assert sb.image_tag is None
        assert sb.container_id is None
        assert sb.status == "stopped"
    finally:
        db.close()


def test_full_creation_to_deletion_flow(client, tmp_path):
    """
    Simulates the complete lifecycle:
      1. Create → queued
      2. Worker updates status → running (simulated via direct DB write)
      3. Stop → stopped
      4. Delete → gone
    """
    # Step 1: Create
    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        r = client.post("/api/sandbox/", json={
            "env_name": "lifecycle_env",
            "description": "full lifecycle test",
            "ttl_days": 1,
        })
    assert r.status_code == 202

    status = client.get("/api/sandbox/lifecycle_env").json()["status"]
    assert status == "queued"

    # Step 2: Simulate worker completing (update DB directly)
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "lifecycle_env")
        sb.status = "running"
        sb.container_id = "fake-container-id"
        sb.container_port = 9001
        sb.image_tag = "forge-lifecycle_env:latest"
        db.commit()
    finally:
        db.close()

    # GET /api/sandbox/{name} cross-checks Docker when status=running;
    # mock docker.from_env so it appears the container is actually running.
    mock_container = MagicMock()
    mock_container.status = "running"
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container
    with patch("docker.from_env", return_value=mock_docker_client):
        status = client.get("/api/sandbox/lifecycle_env").json()["status"]
    assert status == "running"

    # Step 3: Stop
    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.stop = MagicMock()
        resp = client.post("/api/sandbox/lifecycle_env/stop")
    assert resp.status_code == 204

    # Step 4: Delete
    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.remove = MagicMock()
        resp = client.delete("/api/sandbox/lifecycle_env")
    assert resp.status_code == 204

    assert client.get("/api/sandbox/lifecycle_env").status_code == 404


# ---------------------------------------------------------------------------
# env_name validation
# ---------------------------------------------------------------------------

def test_create_sandbox_rejects_name_with_spaces(client):
    """POST /api/sandbox/ returns 422 when env_name contains spaces."""
    resp = client.post("/api/sandbox/", json={
        "env_name": "cli example",
        "env_type": "cli",
        "ttl_days": 7,
    })
    assert resp.status_code == 422
    body = resp.json()
    assert any("env_name" in str(e.get("loc", "")) for e in body["detail"])


def test_create_sandbox_rejects_name_starting_with_hyphen(client):
    """env_name must start with a letter or digit."""
    resp = client.post("/api/sandbox/", json={
        "env_name": "-badname",
        "ttl_days": 7,
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# CLI environment
# ---------------------------------------------------------------------------

def test_create_cli_sandbox_queues_successfully(client):
    """POST /api/sandbox/ with env_type=cli creates a DB row with status=queued."""
    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        resp = client.post("/api/sandbox/", json={
            "env_name": "cli_env",
            "env_type": "cli",
            "ttl_days": 7,
        })
    assert resp.status_code == 202
    body = resp.json()
    assert body["env_name"] == "cli_env"

    info = client.get("/api/sandbox/cli_env").json()
    assert info["status"] == "queued"
    assert info["env_type"] == "cli"


def test_cli_sandbox_full_lifecycle(client):
    """CLI sandbox: queued → running (builtin:cli) → stop → delete."""
    # Create
    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        r = client.post("/api/sandbox/", json={
            "env_name": "cli_lifecycle",
            "env_type": "cli",
            "ttl_days": 1,
        })
    assert r.status_code == 202

    # Simulate worker completing (CLI has no port)
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "cli_lifecycle")
        sb.status = "running"
        sb.container_id = "fake-cli-container-id"
        sb.container_port = None
        sb.image_tag = "builtin:cli"
        db.commit()
    finally:
        db.close()

    mock_container = MagicMock()
    mock_container.status = "running"
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container
    with patch("docker.from_env", return_value=mock_docker_client):
        info = client.get("/api/sandbox/cli_lifecycle").json()
    assert info["status"] == "running"
    assert info["env_type"] == "cli"

    # Stop
    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.stop = MagicMock()
        resp = client.post("/api/sandbox/cli_lifecycle/stop")
    assert resp.status_code == 204

    # Delete
    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.remove = MagicMock()
        resp = client.delete("/api/sandbox/cli_lifecycle")
    assert resp.status_code == 204

    assert client.get("/api/sandbox/cli_lifecycle").status_code == 404


def test_cli_sandbox_start_restarts_container(client):
    """POST /api/sandbox/{name}/start on a stopped CLI env calls ContainerRuntime.start()."""
    _add_sandbox(client, "cli_stopped", status="stopped")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "cli_stopped")
        sb.container_id = "fake-cli-id"
        sb.image_tag = "builtin:cli"
        db.commit()
    finally:
        db.close()

    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.start.return_value = ("fake-cli-id", 0)
        resp = client.post("/api/sandbox/cli_stopped/start")

    assert resp.status_code == 200

    # GET cross-checks Docker when status=running; mock so it doesn't reset status
    mock_container = MagicMock()
    mock_container.status = "running"
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container
    with patch("docker.from_env", return_value=mock_docker_client):
        info = client.get("/api/sandbox/cli_stopped").json()
    assert info["status"] == "running"


# ---------------------------------------------------------------------------
# Browser environment
# ---------------------------------------------------------------------------

def test_create_browser_sandbox_queues_successfully(client):
    """POST /api/sandbox/ with env_type=browser creates a DB row with status=queued."""
    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        resp = client.post("/api/sandbox/", json={
            "env_name": "browser_env",
            "env_type": "browser",
            "ttl_days": 7,
        })
    assert resp.status_code == 202
    body = resp.json()
    assert body["env_name"] == "browser_env"

    info = client.get("/api/sandbox/browser_env").json()
    assert info["status"] == "queued"
    assert info["env_type"] == "browser"


def test_browser_sandbox_full_lifecycle(client):
    """Browser sandbox: queued → running (builtin:browser, with port) → stop → delete."""
    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        r = client.post("/api/sandbox/", json={
            "env_name": "browser_lifecycle",
            "env_type": "browser",
            "ttl_days": 1,
        })
    assert r.status_code == 202

    # Simulate worker completing (browser has VNC port)
    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "browser_lifecycle")
        sb.status = "running"
        sb.container_id = "fake-browser-container-id"
        sb.container_port = 33001
        sb.image_tag = "builtin:browser"
        db.commit()
    finally:
        db.close()

    mock_container = MagicMock()
    mock_container.status = "running"
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container
    with patch("docker.from_env", return_value=mock_docker_client):
        info = client.get("/api/sandbox/browser_lifecycle").json()
    assert info["status"] == "running"
    assert info["env_type"] == "browser"
    assert info["container_port"] == 33001

    # Stop
    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.stop = MagicMock()
        resp = client.post("/api/sandbox/browser_lifecycle/stop")
    assert resp.status_code == 204

    # Delete
    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.remove = MagicMock()
        resp = client.delete("/api/sandbox/browser_lifecycle")
    assert resp.status_code == 204

    assert client.get("/api/sandbox/browser_lifecycle").status_code == 404


def test_browser_sandbox_start_restarts_container(client):
    """POST /api/sandbox/{name}/start on a stopped browser env calls ContainerRuntime.start()."""
    _add_sandbox(client, "browser_stopped", status="stopped")

    from backend.app import database
    db: Session = database.get_session_factory()()
    try:
        sb = db.get(SandboxEnvironment, "browser_stopped")
        sb.container_id = "fake-browser-id"
        sb.image_tag = "builtin:browser"
        db.commit()
    finally:
        db.close()

    with patch("forge.envgen.container.ContainerRuntime") as mock_rt:
        mock_rt.return_value.start.return_value = ("fake-browser-id", 33001)
        resp = client.post("/api/sandbox/browser_stopped/start")

    assert resp.status_code == 200

    # GET cross-checks Docker when status=running; mock so it doesn't reset status
    mock_container = MagicMock()
    mock_container.status = "running"
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.return_value = mock_container
    with patch("docker.from_env", return_value=mock_docker_client):
        info = client.get("/api/sandbox/browser_stopped").json()
    assert info["status"] == "running"
    assert info["container_port"] == 33001


# ---------------------------------------------------------------------------
# Celery task worker path — CLI
# ---------------------------------------------------------------------------

def test_build_cli_task_pulls_image_and_creates_container(client):
    """
    build_sandbox_task with env_type=cli:
    - pulls ubuntu:22.04 via subprocess (not the SDK, which hangs on credential helpers)
    - creates the container via Docker SDK
    - publishes progress messages and a final done:true signal to Redis
    - updates DB status to running with image_tag=builtin:cli
    """
    import json
    import docker.errors

    # Create the DB record (status=queued)
    mocks = _mock_create_deps()
    with mocks[0], mocks[1]:
        r = client.post("/api/sandbox/", json={
            "env_name": "cli_task_test",
            "env_type": "cli",
            "ttl_days": 1,
        })
    assert r.status_code == 202

    # Capture Redis publish calls
    published: list[dict] = []
    mock_redis = MagicMock()
    mock_redis.publish.side_effect = lambda _ch, data: published.append(json.loads(data))

    # Mock Docker: _remove_existing raises NotFound, containers.run returns a container
    mock_container = MagicMock()
    mock_container.id = "fake-cli-abc123"
    mock_docker_client = MagicMock()
    mock_docker_client.containers.get.side_effect = docker.errors.NotFound("not found")
    mock_docker_client.containers.run.return_value = mock_container

    with patch("redis.from_url", return_value=mock_redis), \
         patch("forge.envgen.container.subprocess.run") as mock_subproc, \
         patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container.docker.from_env", return_value=mock_docker_client):

        mock_subproc.return_value = MagicMock(returncode=0)

        from backend.app.worker.tasks import build_sandbox_task
        build_sandbox_task(
            job_id="test-job-cli",
            env_name="cli_task_test",
            env_type="cli",
        )

    # subprocess called to pull ubuntu:22.04 (not the SDK), with per-attempt timeout
    mock_subproc.assert_called_once_with(
        ["docker", "pull", "ubuntu:22.04"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Docker SDK used to run the container
    mock_docker_client.containers.run.assert_called_once()
    run_kwargs = mock_docker_client.containers.run.call_args.kwargs
    assert run_kwargs["image"] == "ubuntu:22.04"
    assert run_kwargs["detach"] is True
    assert run_kwargs["command"] == ["tail", "-f", "/dev/null"]

    # Progress messages published to Redis
    log_texts = [m["log"] for m in published if "log" in m]
    assert any("cli_task_test" in msg for msg in log_texts)
    assert any("ready" in msg.lower() for msg in log_texts)
    assert any(m.get("done") is True for m in published)

    # DB updated to running with builtin:cli tag
    mock_running = MagicMock()
    mock_running.status = "running"
    mock_docker_for_get = MagicMock()
    mock_docker_for_get.containers.get.return_value = mock_running
    with patch("docker.from_env", return_value=mock_docker_for_get):
        info = client.get("/api/sandbox/cli_task_test").json()
    assert info["status"] == "running"
    assert info["env_type"] == "cli"
