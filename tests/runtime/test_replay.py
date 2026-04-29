# tests/runtime/test_replay.py
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models import Episode, EpisodeStep
from forge.runtime.replay import ReplayService


def make_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def insert_episode_with_steps(db, n_steps: int, episode_id: str = "ep_00000001") -> str:
    ep = Episode(
        id=episode_id,
        env_name="test_env",
        task_name="test_task",
        seed=1,
        agent_id="random_policy",
        status="completed",
        total_steps=n_steps,
        total_reward=0.5 * n_steps,
        passed=False,
        started_at=datetime.now(timezone.utc),
    )
    db.add(ep)
    for i in range(n_steps):
        db.add(EpisodeStep(
            episode_id=episode_id,
            step_index=i,
            action=f'{{"type": "action_{i}"}}',
            reward=0.5,
            verifier_results="[]",
            diff="{}",
            events="[]",
            state_hash_before="abc",
            state_hash_after="def",
            terminated=False,
            truncated=(i == n_steps - 1),
        ))
    db.commit()
    return episode_id


def test_load_episode_returns_steps_in_order():
    db = make_db()
    insert_episode_with_steps(db, 3)
    record = ReplayService().load_episode("ep_00000001", db)
    assert [s.step_index for s in record.steps] == [0, 1, 2]


def test_load_episode_returns_episode_metadata():
    db = make_db()
    insert_episode_with_steps(db, 2)
    record = ReplayService().load_episode("ep_00000001", db)
    assert record.episode.env_name == "test_env"
    assert record.episode.total_steps == 2


def test_branch_from_returns_n_actions():
    db = make_db()
    insert_episode_with_steps(db, 5)
    actions = ReplayService().branch_from("ep_00000001", 3, db)
    assert len(actions) == 3
    assert all(isinstance(a, dict) for a in actions)


def test_branch_from_returns_correct_action_sequence():
    db = make_db()
    insert_episode_with_steps(db, 5)
    actions = ReplayService().branch_from("ep_00000001", 2, db)
    assert actions[0] == {"type": "action_0"}
    assert actions[1] == {"type": "action_1"}


def test_branch_from_step_0_returns_empty():
    db = make_db()
    insert_episode_with_steps(db, 3)
    actions = ReplayService().branch_from("ep_00000001", 0, db)
    assert actions == []


def test_load_episode_raises_for_missing_id():
    import pytest
    db = make_db()
    with pytest.raises(ValueError, match="not found"):
        ReplayService().load_episode("nonexistent", db)
