from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.app.worker import env_artifacts, rollout_tasks, tasks
from forge.schema.state_schema import StateSchemaManifest

MANIFEST = StateSchemaManifest(env_name="mail", fields={"inbox": {"type": "array"}})


def test_load_manifest_returns_none_when_absent(tmp_path):
    assert env_artifacts.load_manifest(tmp_path) is None


def test_load_manifest_parses_the_schema(tmp_path):
    (tmp_path / "state_schema.json").write_text(MANIFEST.model_dump_json())
    assert env_artifacts.load_manifest(tmp_path) == MANIFEST


def test_load_manifest_reports_a_corrupt_schema(tmp_path, caplog):
    (tmp_path / "state_schema.json").write_text("{broken")
    with caplog.at_level(logging.WARNING):
        assert env_artifacts.load_manifest(tmp_path) is None
    assert "state_schema.json" in caplog.text


@pytest.fixture
def worker_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/worker.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "envs"))
    from backend.app import database
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    database.init_db()
    env_dir = tmp_path / "envs" / "mail"
    env_dir.mkdir(parents=True)
    (env_dir / "state_schema.json").write_text(MANIFEST.model_dump_json())
    (env_dir / "port").write_text("9100")
    return tmp_path


# ---------------------------------------------------------------------------
# Rollout episodes: status, failure classification and the job counter
# ---------------------------------------------------------------------------

def _add_rollout_job(job_id: str, num_episodes: int) -> None:
    from backend.app import database
    from backend.app.models import RolloutJob

    with database.get_session_factory()() as db:
        db.add(RolloutJob(
            id=job_id, env_name="mail", task_name="triage", agent_id="random",
            num_episodes=num_episodes, seed_start=0, status="running",
            episodes_completed=0, created_at=datetime.now(timezone.utc),
        ))
        db.commit()


def _rollout_state(job_id: str):
    from backend.app import database
    from backend.app.models import Episode, RolloutJob

    with database.get_session_factory()() as db:
        job = db.get(RolloutJob, job_id)
        episodes = db.query(Episode).filter_by(env_name="mail").all()
        return (job.status, job.episodes_completed), [
            (ep.status, ep.failure_type, ep.attempts_json is not None) for ep in episodes
        ]


def _env_declaring(*task_ids: str):
    """A stand-in env builder whose task source declares `task_ids`."""
    from forge.runtime.task_source import StaticTaskSource

    env = SimpleNamespace(task_source=StaticTaskSource([{"id": t, "objective": f"do {t}"} for t in task_ids]))
    return lambda env_name: (lambda: env)


@pytest.fixture
def declares_triage(monkeypatch):
    monkeypatch.setattr(rollout_tasks, "forge_env_builder", _env_declaring("triage", "archive"))


def test_rollout_episode_crash_is_classified_and_completes_the_job(worker_db, declares_triage):
    _add_rollout_job("rj_crash", num_episodes=1)

    with patch(
        "forge.runtime.reliability.execute_reliable_episode",
        side_effect=RuntimeError("connection refused"),
    ):
        tasks.run_episode_task.apply(args=["rj_crash", 0, 7])

    job, episodes = _rollout_state("rj_crash")
    assert job == ("completed", 1)
    assert episodes == [("failed", "infrastructure", False)]


def test_rollout_episode_that_hits_its_budget_is_truncated(worker_db, declares_triage):
    _add_rollout_job("rj_budget", num_episodes=2)
    result = SimpleNamespace(termination_reason="max_steps")

    with patch("forge.runtime.reliability.execute_reliable_episode", return_value=(result, [])):
        tasks.run_episode_task.apply(args=["rj_budget", 0, 1])

    job, episodes = _rollout_state("rj_budget")
    assert job == ("running", 1)
    assert episodes == [("truncated", None, True)]


def test_rollout_episode_that_submits_keeps_its_running_status(worker_db, declares_triage):
    # False-positive guard: only budget reasons truncate.
    _add_rollout_job("rj_done", num_episodes=1)
    result = SimpleNamespace(termination_reason="submitted")

    with patch("forge.runtime.reliability.execute_reliable_episode", return_value=(result, [])):
        tasks.run_episode_task.apply(args=["rj_done", 0, 1])

    job, episodes = _rollout_state("rj_done")
    assert job == ("completed", 1)
    assert episodes == [("running", None, True)]


def test_rollout_dispatch_failure_marks_the_job_failed(worker_db):
    from backend.app import database
    from backend.app.models import RolloutJob

    _add_rollout_job("rj_broker", num_episodes=2)
    with patch.object(rollout_tasks, "group", side_effect=RuntimeError("broker down")):
        tasks.run_rollout_task.apply(args=["rj_broker"])

    with database.get_session_factory()() as db:
        job = db.get(RolloutJob, "rj_broker")
        assert (job.status, job.error) == ("failed", "broker down")
        assert job.completed_at is not None


def test_every_task_keeps_its_registered_queue_name():
    # Jobs already in the broker route by name, so moving a task between
    # modules must not rename it.
    from backend.app.worker.celery_app import celery

    expected = {
        f"backend.app.worker.tasks.{name}" for name in tasks.__all__
    }
    assert expected <= set(celery.tasks)
    assert {getattr(tasks, name).name for name in tasks.__all__} == expected


def _rollout_failure(job_id: str):
    from backend.app import database
    from backend.app.models import Episode

    with database.get_session_factory()() as db:
        episode = db.query(Episode).filter_by(env_name="mail").one()
        return episode.status, episode.failure_type, episode.failure_reason, episode.task_name


def test_a_rollout_for_an_undeclared_task_fails_instead_of_running_another(worker_db, monkeypatch):
    monkeypatch.setattr(rollout_tasks, "forge_env_builder", _env_declaring("archive", "reply"))
    _add_rollout_job("rj_unknown", num_episodes=1)

    with patch("forge.runtime.reliability.execute_reliable_episode") as execute:
        tasks.run_episode_task.apply(args=["rj_unknown", 0, 1])

    execute.assert_not_called()
    status, failure_type, reason, task_name = _rollout_failure("rj_unknown")
    assert (status, failure_type, task_name) == ("failed", "configuration", "triage")
    assert "triage" in reason and "archive, reply" in reason
    job, _ = _rollout_state("rj_unknown")
    assert job == ("completed", 1)


def test_a_declared_task_is_the_one_the_attempt_runs():
    env = _env_declaring("triage", "archive")("mail")()

    assert rollout_tasks._select_task(env, "archive").id == "archive"


def test_selecting_an_undeclared_task_raises_rather_than_substituting():
    env = _env_declaring("archive")("mail")()

    with pytest.raises(rollout_tasks.UnknownTaskError, match="'triage'.*archive"):
        rollout_tasks._select_task(env, "triage")


def test_an_env_without_declared_tasks_runs_the_name_as_its_objective():
    # Nothing is substituted: the episode runs exactly what was asked for.
    env = _env_declaring()("mail")()

    assert rollout_tasks._select_task(env, "file the report") == {
        "id": "file the report", "objective": "file the report",
    }


def test_an_env_that_fails_to_build_is_still_classified_by_the_retry_path(worker_db, monkeypatch):
    def broken_builder(env_name):
        raise RuntimeError("Environment 'mail' violates network policy")

    monkeypatch.setattr(rollout_tasks, "forge_env_builder", broken_builder)
    _add_rollout_job("rj_broken", num_episodes=1)

    with patch(
        "forge.runtime.reliability.execute_reliable_episode",
        side_effect=RuntimeError("connection refused"),
    ) as execute:
        tasks.run_episode_task.apply(args=["rj_broken", 0, 1])

    execute.assert_called_once()
    job, episodes = _rollout_state("rj_broken")
    assert job == ("completed", 1)
    assert episodes == [("failed", "infrastructure", False)]
