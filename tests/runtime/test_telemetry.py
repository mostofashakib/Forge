# tests/runtime/test_telemetry.py
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models import Episode, EpisodeStep
from forge.runtime.snapshot import StepSnapshot


def make_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    # Pre-create Episode row (as RunnerService would before constructing TelemetryClient)
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
    return db


def make_snapshot(step_index: int = 0, terminated: bool = False) -> StepSnapshot:
    return StepSnapshot(
        episode_id="ep_00000001",
        step_index=step_index,
        state_hash_before="abc",
        state_hash_after="def",
        action={"type": "increment"},
        events=[],
        reward=0.5,
        verifier_results=[],
        diff={"added": {}, "changed": {}, "removed": {}},
        terminated=terminated,
        truncated=False,
    )


def make_client(db, jsonl_path=None):
    from forge.runtime.telemetry import TelemetryClient
    return TelemetryClient(
        episode_id="ep_00000001",
        db_session=db,
        jsonl_path=jsonl_path,
    )


def test_record_step_writes_sqlite_row():
    db = make_db()
    client = make_client(db)
    client.record_step(make_snapshot())
    step = db.query(EpisodeStep).filter_by(episode_id="ep_00000001").first()
    assert step is not None
    assert step.step_index == 0
    assert json.loads(step.action) == {"type": "increment"}
    assert step.reward == 0.5
    assert step.state_hash_before == "abc"


def test_record_step_writes_jsonl_line(tmp_path):
    db = make_db()
    jsonl_path = tmp_path / "ep_00000001.jsonl"
    client = make_client(db, jsonl_path=jsonl_path)
    client.record_step(make_snapshot())
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["episode_id"] == "ep_00000001"
    assert data["step_index"] == 0


def test_record_step_appends_multiple_jsonl_lines(tmp_path):
    db = make_db()
    jsonl_path = tmp_path / "ep_00000001.jsonl"
    client = make_client(db, jsonl_path=jsonl_path)
    client.record_step(make_snapshot(step_index=0))
    client.record_step(make_snapshot(step_index=1))
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_record_step_no_jsonl_path_does_not_raise():
    db = make_db()
    client = make_client(db, jsonl_path=None)
    client.record_step(make_snapshot())  # should not raise


def test_complete_episode_updates_row():
    db = make_db()
    client = make_client(db)
    client.complete_episode(total_reward=1.5, passed=True, total_steps=7)
    ep = db.get(Episode, "ep_00000001")
    assert ep.status == "completed"
    assert ep.total_reward == 1.5
    assert ep.passed is True
    assert ep.total_steps == 7
    assert ep.completed_at is not None


def test_complete_episode_failed_sets_passed_false():
    db = make_db()
    client = make_client(db)
    client.complete_episode(total_reward=-0.5, passed=False, total_steps=50)
    ep = db.get(Episode, "ep_00000001")
    assert ep.status == "completed"
    assert ep.passed is False
