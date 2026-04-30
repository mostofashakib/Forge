import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from backend.app.database import Base
from backend.app.models import AuditLog
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


def test_audit_list_empty_returns_list(api_client):
    resp = api_client.get("/api/audit/?env_name=nonexistent_env_xyz")
    assert resp.status_code == 200
    assert resp.json() == []


def test_audit_missing_env_name_returns_422(api_client):
    resp = api_client.get("/api/audit/")
    assert resp.status_code == 422


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_audit_log_model(db):
    log = AuditLog(
        episode_id="ep_00000001",
        step_index=3,
        actor="agent",
        action_type="offer_refund",
        rule_id="no_refund_without_order",
        violation="Refund attempted without valid order",
        severity="high",
        created_at=datetime.now(timezone.utc),
    )
    db.add(log)
    db.commit()
    fetched = db.get(AuditLog, 1)
    assert fetched.episode_id == "ep_00000001"
    assert fetched.rule_id == "no_refund_without_order"
    assert fetched.severity == "high"
    assert fetched.actor == "agent"
