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


# ---------------------------------------------------------------------------
# Statistical detection
# ---------------------------------------------------------------------------

def test_episode_features_are_extracted_from_recorded_steps():
    from backend.app.api.detect import _episode_features

    class _Ep:
        id = "cep_abc"
        total_steps = 3
        total_reward = 0.5

    features = _episode_features(
        [_Ep()],
        {"cep_abc": [{"command": "ls -la"}, {"command": "cat x"}, {"command": "ls -la"}]},
    )

    assert features[0].episode_id == "cep_abc"
    assert features[0].reward == 0.5
    # The command verb is the action, so "ls -la" and "ls foo" are one action.
    assert features[0].actions == ["ls", "cat", "ls"]


def test_episode_features_tolerate_missing_steps():
    from backend.app.api.detect import _episode_features

    class _Ep:
        id = "cep_none"
        total_steps = 0
        total_reward = 0.0

    assert _episode_features([_Ep()], {})[0].actions == []


def test_a_step_without_a_command_contributes_no_action():
    from backend.app.api.detect import _episode_features

    class _Ep:
        id = "cep_x"
        total_steps = 1
        total_reward = 0.1

    features = _episode_features([_Ep()], {"cep_x": [{"exit_code": 0}]})

    assert features[0].actions == []
