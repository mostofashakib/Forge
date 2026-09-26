from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.app.worker import tasks
from forge.schema.state_schema import StateSchemaManifest

MANIFEST = StateSchemaManifest(env_name="mail", fields={"inbox": {"type": "array"}})


def test_load_manifest_returns_none_when_absent(tmp_path):
    assert tasks._load_manifest(tmp_path) is None


def test_load_manifest_parses_the_schema(tmp_path):
    (tmp_path / "state_schema.json").write_text(MANIFEST.model_dump_json())
    assert tasks._load_manifest(tmp_path) == MANIFEST


def test_load_manifest_reports_a_corrupt_schema(tmp_path, caplog):
    (tmp_path / "state_schema.json").write_text("{broken")
    with caplog.at_level(logging.WARNING):
        assert tasks._load_manifest(tmp_path) is None
    assert "state_schema.json" in caplog.text


@pytest.fixture
def benchmark_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/bench.db")
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "envs"))
    from backend.app import database
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    database.init_db()
    from backend.app.models import BenchmarkRun
    with database.get_session_factory()() as db:
        db.add(BenchmarkRun(
            id="bm_1", status="queued", domains="mail", depth=1, seeds=3,
            output_dir=str(tmp_path / "out"), created_at=datetime.now(timezone.utc),
        ))
        db.commit()
    env_dir = tmp_path / "envs" / "mail"
    env_dir.mkdir(parents=True)
    (env_dir / "state_schema.json").write_text(MANIFEST.model_dump_json())
    (env_dir / "port").write_text("9100")
    return tmp_path


def test_benchmark_parses_each_manifest_once(benchmark_db):
    task = SimpleNamespace(domain="mail", name="triage", objective="o")

    class _Collector:
        def __init__(self, *_args, **_kwargs):
            pass

        def pending_runs(self, _checkpoint):
            return [{"task": task, "seed": seed} for seed in range(3)]

        def collect(self, run_episode):
            for seed in range(3):
                run_episode(task, seed, benchmark_db / f"ep{seed}.jsonl")

    runner = MagicMock()
    runner.__enter__.return_value.run_episode.return_value = SimpleNamespace(
        total_reward=1.0, termination_reason="done"
    )
    quality = SimpleNamespace(
        env_name="mail", state_coverage_score=1.0, reward_density=0.0,
        dead_end_rate=0.0, action_diversity=0.0, num_episodes=3, num_steps=3,
    )
    parse = MagicMock(wraps=StateSchemaManifest.model_validate_json)

    with patch("redis.from_url"), \
         patch("forge.benchmark.data_collector.DataCollector", _Collector), \
         patch("forge.benchmark.compiled_tasks.CompiledTaskProvider"), \
         patch("forge.envgen.episode_runner.ContainerEpisodeRunner", return_value=runner), \
         patch("forge.envgen.agents.container_agent.make_container_agent"), \
         patch("forge.benchmark.env_quality.compute_env_quality", return_value=quality), \
         patch("forge.benchmark.report.BenchmarkReport"), \
         patch.object(StateSchemaManifest, "model_validate_json", parse):
        tasks.run_benchmark_task.apply(
            args=["bm_1", ["mail"], 1, 3, str(benchmark_db / "out")]
        )

    from backend.app import database
    from backend.app.models import BenchmarkRun
    with database.get_session_factory()() as db:
        assert db.get(BenchmarkRun, "bm_1").status == "done"
    assert runner.__enter__.return_value.run_episode.call_count == 3
    assert parse.call_count == 1


def test_benchmark_survives_and_logs_a_failing_progress_publish(benchmark_db, caplog):
    redis_client = MagicMock()
    redis_client.publish.side_effect = ConnectionError("redis went away")

    with patch("redis.from_url", return_value=redis_client), \
         patch("forge.benchmark.compiled_tasks.CompiledTaskProvider"), \
         patch("forge.benchmark.data_collector.DataCollector") as collector, \
         patch("forge.benchmark.report.BenchmarkReport"), \
         caplog.at_level(logging.DEBUG, logger="backend.app.worker.tasks"):
        collector.return_value.pending_runs.return_value = []
        tasks.run_benchmark_task.apply(args=["bm_1", ["mail"], 1, 3, str(benchmark_db / "out")])

    from backend.app import database
    from backend.app.models import BenchmarkRun
    with database.get_session_factory()() as db:
        assert db.get(BenchmarkRun, "bm_1").status == "done"
    assert "progress publish failed" in caplog.text


def test_benchmark_episodes_hold_their_environment_lock(benchmark_db):
    # A benchmark drives the same container an agent run might be using, so
    # it takes the same per-environment lock around every episode.
    from contextlib import contextmanager

    task = SimpleNamespace(domain="mail", name="triage", objective="o")
    held: list[str] = []
    seen_while_running: list[list[str]] = []

    @contextmanager
    def fake_lock(_redis_client, env_name):
        held.append(env_name)
        try:
            yield
        finally:
            held.remove(env_name)

    class _Collector:
        def __init__(self, *_args, **_kwargs):
            pass

        def pending_runs(self, _checkpoint):
            return [{"task": task, "seed": 0}]

        def collect(self, run_episode):
            run_episode(task, 0, benchmark_db / "ep0.jsonl")

    runner = MagicMock()
    runner.__enter__.return_value.run_episode.side_effect = (
        lambda *a, **k: seen_while_running.append(list(held))
        or SimpleNamespace(total_reward=0.0, termination_reason="done")
    )
    quality = SimpleNamespace(
        env_name="mail", state_coverage_score=1.0, reward_density=0.0,
        dead_end_rate=0.0, action_diversity=0.0, num_episodes=1, num_steps=1,
    )
    with patch.object(tasks, "exclusive_environment", fake_lock), \
         patch("redis.from_url"), \
         patch("forge.benchmark.data_collector.DataCollector", _Collector), \
         patch("forge.benchmark.compiled_tasks.CompiledTaskProvider"), \
         patch("forge.envgen.episode_runner.ContainerEpisodeRunner", return_value=runner), \
         patch("forge.envgen.agents.container_agent.make_container_agent"), \
         patch("forge.benchmark.env_quality.compute_env_quality", return_value=quality), \
         patch("forge.benchmark.report.BenchmarkReport"):
        tasks.run_benchmark_task.apply(
            args=["bm_1", ["mail"], 1, 1, str(benchmark_db / "out")]
        )

    assert seen_while_running == [["mail"]]
    assert held == []
