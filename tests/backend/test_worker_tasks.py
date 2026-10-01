from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.app.worker import benchmark_tasks, env_artifacts, rollout_tasks, tasks
from forge.benchmark.env_quality import EnvQualityMetrics
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
    quality = EnvQualityMetrics(
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
    quality = EnvQualityMetrics(
        env_name="mail", state_coverage_score=1.0, reward_density=0.0,
        dead_end_rate=0.0, action_diversity=0.0, num_episodes=1, num_steps=1,
    )
    with patch.object(benchmark_tasks, "exclusive_environment", fake_lock), \
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


def test_rollout_episode_crash_is_classified_and_completes_the_job(benchmark_db):
    _add_rollout_job("rj_crash", num_episodes=1)

    with patch(
        "forge.runtime.reliability.execute_reliable_episode",
        side_effect=RuntimeError("connection refused"),
    ):
        tasks.run_episode_task.apply(args=["rj_crash", 0, 7])

    job, episodes = _rollout_state("rj_crash")
    assert job == ("completed", 1)
    assert episodes == [("failed", "infrastructure", False)]


def test_rollout_episode_that_hits_its_budget_is_truncated(benchmark_db):
    _add_rollout_job("rj_budget", num_episodes=2)
    result = SimpleNamespace(termination_reason="max_steps")

    with patch("forge.runtime.reliability.execute_reliable_episode", return_value=(result, [])):
        tasks.run_episode_task.apply(args=["rj_budget", 0, 1])

    job, episodes = _rollout_state("rj_budget")
    assert job == ("running", 1)
    assert episodes == [("truncated", None, True)]


def test_rollout_episode_that_submits_keeps_its_running_status(benchmark_db):
    # False-positive guard: only budget reasons truncate.
    _add_rollout_job("rj_done", num_episodes=1)
    result = SimpleNamespace(termination_reason="submitted")

    with patch("forge.runtime.reliability.execute_reliable_episode", return_value=(result, [])):
        tasks.run_episode_task.apply(args=["rj_done", 0, 1])

    job, episodes = _rollout_state("rj_done")
    assert job == ("completed", 1)
    assert episodes == [("running", None, True)]


def test_rollout_dispatch_failure_marks_the_job_failed(benchmark_db):
    from backend.app import database
    from backend.app.models import RolloutJob

    _add_rollout_job("rj_broker", num_episodes=2)
    with patch.object(rollout_tasks, "group", side_effect=RuntimeError("broker down")):
        tasks.run_rollout_task.apply(args=["rj_broker"])

    with database.get_session_factory()() as db:
        job = db.get(RolloutJob, "rj_broker")
        assert (job.status, job.error) == ("failed", "broker down")
        assert job.completed_at is not None


def test_evaluation_fails_the_run_when_redis_is_unreachable(benchmark_db):
    from backend.app import database
    from backend.app.models import BenchmarkRun

    redis_client = MagicMock()
    redis_client.ping.side_effect = ConnectionError("no redis")
    with patch("redis.from_url", return_value=redis_client):
        tasks.run_evaluation_task("bm_1", "forge", {})

    with database.get_session_factory()() as db:
        run = db.get(BenchmarkRun, "bm_1")
        assert (run.status, run.error) == ("failed", "no redis")


def test_evaluation_rejects_an_unknown_engine(benchmark_db):
    from backend.app import database
    from backend.app.models import BenchmarkRun

    redis_client = MagicMock()
    with patch("redis.from_url", return_value=redis_client):
        tasks.run_evaluation_task("bm_1", "nope", {})

    with database.get_session_factory()() as db:
        run = db.get(BenchmarkRun, "bm_1")
        assert run.status == "failed"
        assert "unsupported evaluation engine" in run.error
    published = [c.args[1] for c in redis_client.publish.call_args_list]
    assert any("unsupported evaluation engine" in m for m in published)


def test_every_task_keeps_its_registered_queue_name():
    # Jobs already in the broker route by name, so moving a task between
    # modules must not rename it.
    from backend.app.worker.celery_app import celery

    expected = {
        f"backend.app.worker.tasks.{name}" for name in tasks.__all__
    }
    assert expected <= set(celery.tasks)
    assert {getattr(tasks, name).name for name in tasks.__all__} == expected


# ---------------------------------------------------------------------------
# Transfer evaluation reports only what an evaluator measured
# ---------------------------------------------------------------------------

def _benchmark_run(run_id: str):
    from backend.app import database
    from backend.app.models import BenchmarkRun

    with database.get_session_factory()() as db:
        run = db.get(BenchmarkRun, run_id)
        return run.status, run.error, run.report_json


def test_production_transfer_without_training_data_fails_without_metrics(benchmark_db, tmp_path):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "train_envs: [train_a]\nheldout_envs: [held_a]\n"
        "reward_preset: full_layered_partial\nbase_model: m\nseeds: [0, 1]\n",
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    redis_client = MagicMock()
    with patch("redis.from_url", return_value=redis_client):
        tasks.run_transfer_task("bm_1", {
            "base_model": "m", "seeds": 2, "eval_suite": str(experiment),
            "data_dir": str(tmp_path / "data"), "output_dir": str(tmp_path / "out"),
        })

    status, error, report = _benchmark_run("bm_1")
    assert status == "failed"
    assert "no sft training signal" in error
    assert report is None
    published = [json.loads(c.args[1]) for c in redis_client.publish.call_args_list]
    assert not any("result" in m for m in published)
    assert any("no sft training signal" in m.get("error", "") for m in published)


def test_production_transfer_rejects_a_suite_that_is_not_an_experiment(benchmark_db, tmp_path):
    redis_client = MagicMock()
    with patch("redis.from_url", return_value=redis_client):
        tasks.run_transfer_task("bm_1", {"base_model": "m", "eval_suite": str(tmp_path / "missing.yaml")})

    status, error, report = _benchmark_run("bm_1")
    assert status == "failed"
    assert "experiment config not found" in error
    assert report is None


def test_transfer_reports_the_injected_evaluators_result(benchmark_db):
    from forge.benchmark.transfer_pipeline import TransferResult

    seen = []

    def evaluate(config):
        seen.append(config)
        return TransferResult(
            model_path=config.base_model, eval_suite=config.eval_suite,
            task_completion_rate=0.5, success_at_1=0.25, success_at_3=0.75, num_eval_tasks=8,
        )

    published: list[dict] = []
    with patch("forge.contracts.gpu.GPUDeviceSpec.probe_hardware", return_value={"mps_available": True}):
        benchmark_tasks.execute_transfer_run(
            "bm_1",
            {"base_model": "m", "eval_suite": "s", "seeds": 2, "max_steps": 40, "data_dir": "d"},
            evaluate=evaluate,
            publish=published.append,
        )

    status, error, report = _benchmark_run("bm_1")
    assert (status, error) == ("done", None)
    assert json.loads(report) == {
        "model_path": "m", "eval_suite": "s", "task_completion_rate": 0.5,
        "pass_at_1": 0.25, "pass_at_3": 0.75, "num_eval_tasks": 8,
        "inference_mode": "auto", "device": "Apple Silicon MPS (Metal)",
    }
    assert (seen[0].seeds, seen[0].max_train_steps, str(seen[0].data_dir)) == (2, 40, "d")
    assert seen[0].run_id == "bm_1"
    assert published[-1]["done"] is True
