# tests/backend/test_episode_api.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base


def make_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_episode_model_can_be_created():
    from backend.app.models import Episode
    db = make_memory_db()
    ep = Episode(
        id="ep_00000001",
        env_name="test_env",
        task_name="test_task",
        seed=1,
        agent_id="random_policy",
        status="running",
        total_steps=0,
        total_reward=0.0,
        passed=False,
        started_at=datetime.now(timezone.utc),
    )
    db.add(ep)
    db.commit()
    fetched = db.get(Episode, "ep_00000001")
    assert fetched.env_name == "test_env"
    assert fetched.status == "running"
    db.close()


def test_episode_step_model_can_be_created():
    from backend.app.models import Episode, EpisodeStep
    db = make_memory_db()
    ep = Episode(
        id="ep_00000001",
        env_name="test_env",
        task_name="test_task",
        seed=1,
        agent_id="random_policy",
        status="running",
        total_steps=0,
        total_reward=0.0,
        passed=False,
        started_at=datetime.now(timezone.utc),
    )
    db.add(ep)
    db.flush()
    step = EpisodeStep(
        episode_id="ep_00000001",
        step_index=0,
        action='{"type": "increment"}',
        reward=0.5,
        verifier_results="[]",
        diff="{}",
        events="[]",
        state_hash_before="abc",
        state_hash_after="def",
        terminated=False,
        truncated=False,
    )
    db.add(step)
    db.commit()
    fetched = db.query(EpisodeStep).filter_by(episode_id="ep_00000001").first()
    assert fetched.step_index == 0
    assert fetched.reward == 0.5
    db.close()


# --- append to tests/backend/test_episode_api.py ---

def test_create_episode_inserts_row():
    from backend.app.services import episode_service
    db = make_memory_db()
    ep = episode_service.create_episode(
        episode_id="ep_aabbccdd",
        env_name="my_env",
        task_name="my_task",
        seed=42,
        agent_id="random",
        db=db,
    )
    assert ep.id == "ep_aabbccdd"
    assert ep.status == "running"
    fetched = db.get(__import__("backend.app.models", fromlist=["Episode"]).Episode, "ep_aabbccdd")
    assert fetched is not None
    db.close()


def test_get_episode_returns_none_for_unknown_id():
    from backend.app.services import episode_service
    db = make_memory_db()
    result = episode_service.get_episode("nonexistent", db)
    assert result is None
    db.close()


def test_get_stats_returns_zero_stats_for_empty_env():
    from backend.app.services import episode_service
    db = make_memory_db()
    stats = episode_service.get_stats("empty_env", db)
    assert stats["pass_rate"] == 0.0
    assert stats["avg_reward"] == 0.0
    assert stats["policy_violation_count"] == 0
    assert stats["top_failures"] == []
    db.close()


# --- REST API tests ---
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    return TestClient(app)


@pytest.fixture
def api_client_with_episode(tmp_path, monkeypatch):
    """Client with one completed episode pre-inserted."""
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from backend.app.models import Episode, EpisodeStep
    SessionFactory = database.get_session_factory()
    db = SessionFactory()
    ep = Episode(
        id="ep_0000002a",
        env_name="test_env",
        task_name="test_task",
        seed=42,
        agent_id="random_policy",
        status="completed",
        total_steps=2,
        total_reward=0.8,
        passed=True,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
    )
    db.add(ep)
    for i in range(2):
        db.add(EpisodeStep(
            episode_id="ep_0000002a",
            step_index=i,
            action=f'{{"type": "action_{i}"}}',
            reward=0.4,
            verifier_results="[]",
            diff="{}",
            events="[]",
            state_hash_before="abc",
            state_hash_after="def",
            terminated=(i == 1),
            truncated=False,
        ))
    db.commit()
    db.close()
    return TestClient(app)


def test_post_episodes_returns_episode_id(api_client, monkeypatch):
    async def fake_start_episode(env_name, task_name, seed, agent_id):
        return f"ep_{seed:08x}"
    import backend.app.services.runner_service as rs
    monkeypatch.setattr(rs, "start_episode", fake_start_episode)
    resp = api_client.post("/api/episodes/", json={
        "env_name": "test_env",
        "task_name": "test_task",
        "seed": 1,
        "agent_id": "random_policy",
    })
    assert resp.status_code == 200
    assert resp.json()["episode_id"] == "ep_00000001"


def test_get_episode_returns_full_record(api_client_with_episode):
    resp = api_client_with_episode.get("/api/episodes/ep_0000002a")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "ep_0000002a"
    assert data["status"] == "completed"
    assert len(data["steps"]) == 2


def test_get_episode_returns_404_for_unknown(api_client):
    resp = api_client.get("/api/episodes/ep_unknown")
    assert resp.status_code == 404


def test_list_episodes_filters_by_env_name(api_client_with_episode):
    resp = api_client_with_episode.get("/api/episodes/?env_name=test_env")
    assert resp.status_code == 200
    ids = [ep["id"] for ep in resp.json()]
    assert "ep_0000002a" in ids


def test_branch_returns_action_sequence(api_client_with_episode):
    resp = api_client_with_episode.get("/api/episodes/ep_0000002a/steps/1/branch")
    assert resp.status_code == 200
    actions = resp.json()["actions"]
    assert len(actions) == 1
    assert actions[0] == {"type": "action_0"}


def test_get_env_stats_returns_pass_rate(api_client_with_episode):
    resp = api_client_with_episode.get("/api/envs/test_env/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "pass_rate" in data
    assert data["pass_rate"] == 1.0
    assert "top_failures" in data


def test_get_compiler_input_returns_json(api_client, tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/test.db")
    from backend.app import database
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from backend.app.models import CompileJob
    from forge.extraction.schemas import CompilerInput
    SessionFactory = database.get_session_factory()
    db = SessionFactory()
    ci = CompilerInput(project_name="test_env", domain="test", entities=[], actions=[], tasks=[])
    job = CompileJob(
        id="job_001",
        project_name="test_env",
        status="complete",
        prompt="test",
        compiler_input_json=ci.model_dump_json(),
    )
    db.add(job)
    db.commit()
    db.close()
    resp = api_client.get("/api/envs/test_env/compiler-input")
    assert resp.status_code == 200
    assert resp.json()["project_name"] == "test_env"
