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
