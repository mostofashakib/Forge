import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from backend.app.models import SandboxEnvironment


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from backend.app.main import app
    return TestClient(app)


def test_get_sandbox_not_found(client):
    resp = client.get("/api/sandbox/nonexistent")
    assert resp.status_code == 404


def test_get_sandbox_returns_model(client, tmp_path, monkeypatch):
    from backend.app import database
    db = database.get_session_factory()()
    try:
        db.add(SandboxEnvironment(
            id="my_env",
            status="ready",
            ttl_days=30,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        ))
        db.commit()
    finally:
        db.close()
    resp = client.get("/api/sandbox/my_env")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
