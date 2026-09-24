from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import AgentEpisode, AgentRun
from backend.app.services.episode_sample import load_trajectory_steps, recent_completed_episodes

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _run(db, run_id: str, env: str, minutes: int) -> None:
    db.add(AgentRun(
        id=run_id, env_name=env, agent_id="a", objective=f"goal {run_id}",
        num_episodes=1, created_at=T0 + timedelta(minutes=minutes),
    ))


def _episode(db, ep_id: str, run_id: str, minutes: int, status: str = "completed") -> None:
    db.add(AgentEpisode(
        id=ep_id, run_id=run_id, episode_index=0, seed=0, status=status,
        started_at=T0, completed_at=T0 + timedelta(minutes=minutes),
    ))


def test_samples_completed_episodes_of_the_env_newest_first(db):
    _run(db, "r1", "env", 1)
    _run(db, "other", "other_env", 2)
    _episode(db, "old", "r1", 1)
    _episode(db, "new", "r1", 5)
    _episode(db, "running", "r1", 9, status="running")
    _episode(db, "foreign", "other", 7)
    db.commit()

    sample = recent_completed_episodes("env", db, limit=5)

    assert [ep.id for ep in sample.episodes] == ["new", "old"]
    assert sample.latest_objective == "goal r1"


def test_oldest_first_and_limit(db):
    _run(db, "r1", "env", 1)
    for minute in range(4):
        _episode(db, f"e{minute}", "r1", minute)
    db.commit()

    sample = recent_completed_episodes("env", db, limit=2, oldest_first=True)

    assert [ep.id for ep in sample.episodes] == ["e0", "e1"]


def test_only_the_ten_newest_runs_are_sampled(db):
    for minute in range(11):
        _run(db, f"r{minute}", "env", minute)
        _episode(db, f"e{minute}", f"r{minute}", minute)
    db.commit()

    sample = recent_completed_episodes("env", db, limit=20)

    assert "e0" not in {ep.id for ep in sample.episodes}
    assert len(sample.episodes) == 10


def test_empty_env_has_no_episodes_and_no_objective(db):
    sample = recent_completed_episodes("missing", db, limit=5)
    assert sample.episodes == []
    assert sample.latest_objective is None


def _episode_with_file(tmp_path, lines: list[str]) -> AgentEpisode:
    path = tmp_path / "ep.jsonl"
    path.write_text("\n".join(lines))
    return AgentEpisode(id="ep", jsonl_path=str(path))


def test_steps_skip_the_summary_and_keep_the_last_n(tmp_path):
    lines = [json.dumps({"step_index": i}) for i in range(5)]
    lines.append(json.dumps({"type": "episode_summary"}))
    steps = load_trajectory_steps(_episode_with_file(tmp_path, lines), max_steps=2)
    assert steps == [{"step_index": 3}, {"step_index": 4}]


def test_missing_trajectory_file_yields_no_steps(tmp_path):
    assert load_trajectory_steps(AgentEpisode(id="ep", jsonl_path=str(tmp_path / "nope")), max_steps=5) == []
    assert load_trajectory_steps(AgentEpisode(id="ep", jsonl_path=None), max_steps=5) == []


def test_corrupt_trajectory_is_reported_not_hidden(tmp_path, caplog):
    episode = _episode_with_file(tmp_path, ['{"step_index": 0}', "{broken"])
    with caplog.at_level(logging.WARNING):
        assert load_trajectory_steps(episode, max_steps=5) == []
    assert "ep.jsonl" in caplog.text
