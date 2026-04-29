import json
import pytest
from pathlib import Path
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from backend.app.database import Base
from backend.app.models import Episode, EpisodeStep, ExportJob
from backend.app.services.export_service import run_export, _write_trajectories, _write_rewards


@pytest.fixture
def db_with_episodes():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # 4 episodes: i=0,2 passed (i%2==0), i=1,3 failed
        episodes = [
            Episode(
                id=f"ep_{i:08x}",
                env_name="test_env",
                task_name=f"task_{i % 2}",
                seed=i * 10,
                agent_id="random",
                status="completed",
                total_steps=3,
                total_reward=float(i),
                passed=(i % 2 == 0),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
            for i in range(4)
        ]
        for ep in episodes:
            session.add(ep)
        for ep in episodes:
            for s in range(3):
                step = EpisodeStep(
                    episode_id=ep.id,
                    step_index=s,
                    action=json.dumps({"type": "close_ticket"}),
                    reward=0.1,
                    verifier_results=json.dumps([{"name": "check_a", "passed": ep.passed}]),
                    diff=json.dumps({}),
                    events=json.dumps([]),
                    state_hash_before=f"h{s}",
                    state_hash_after=f"h{s+1}",
                    terminated=(s == 2),
                    truncated=False,
                )
                session.add(step)
        session.commit()
        yield session
    Base.metadata.drop_all(engine)


def test_write_trajectories(db_with_episodes, tmp_path):
    _write_trajectories("test_env", db_with_episodes, tmp_path)
    lines = (tmp_path / "trajectories.jsonl").read_text().strip().split("\n")
    assert len(lines) == 4
    for line in lines:
        obj = json.loads(line)
        assert "episode_id" in obj
        assert "steps" in obj
        assert len(obj["steps"]) == 3


def test_write_rewards(db_with_episodes, tmp_path):
    _write_rewards("test_env", db_with_episodes, tmp_path)
    lines = (tmp_path / "rewards.jsonl").read_text().strip().split("\n")
    assert len(lines) == 4
    for line in lines:
        obj = json.loads(line)
        assert "episode_id" in obj
        assert "total_reward" in obj
        assert "passed" in obj


def test_run_export_all_formats(db_with_episodes, tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.services.export_service.BASE_DIR", tmp_path)
    job = ExportJob(
        id="ex_test0001",
        env_name="test_env",
        formats=json.dumps(["trajectories", "rewards", "verifier_results", "sft_pairs", "preference_pairs", "grpo_rollouts"]),
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    db_with_episodes.add(job)
    db_with_episodes.commit()
    run_export("ex_test0001", db_with_episodes)
    db_with_episodes.refresh(job)
    assert job.status == "completed"
    out = Path(job.output_path)
    assert (out / "trajectories.jsonl").exists()
    assert (out / "rewards.jsonl").exists()
    assert (out / "verifier_results.jsonl").exists()
    assert (out / "sft_pairs.jsonl").exists()
    assert (out / "preference_pairs.jsonl").exists()
    assert (out / "grpo_rollouts.parquet").exists()


def test_sft_pairs_passed_only(db_with_episodes, tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.services.export_service.BASE_DIR", tmp_path)
    job = ExportJob(
        id="ex_sft00001",
        env_name="test_env",
        formats=json.dumps(["sft_pairs"]),
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    db_with_episodes.add(job)
    db_with_episodes.commit()
    run_export("ex_sft00001", db_with_episodes)
    db_with_episodes.refresh(job)
    lines = (Path(job.output_path) / "sft_pairs.jsonl").read_text().strip().split("\n")
    # 2 episodes passed (i=0 and i=2) * 3 steps each = 6 sft pairs
    assert len(lines) == 6


def test_preference_pairs(db_with_episodes, tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.services.export_service.BASE_DIR", tmp_path)
    job = ExportJob(
        id="ex_pref0001",
        env_name="test_env",
        formats=json.dumps(["preference_pairs"]),
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    db_with_episodes.add(job)
    db_with_episodes.commit()
    run_export("ex_pref0001", db_with_episodes)
    db_with_episodes.refresh(job)
    out_path = Path(job.output_path) / "preference_pairs.jsonl"
    lines = [json.loads(l) for l in out_path.read_text().strip().split("\n") if l]
    for pair in lines:
        assert "chosen" in pair
        assert "rejected" in pair
        assert pair["chosen"]["total_reward"] >= pair["rejected"]["total_reward"]
