"""Celery tasks for benchmark collection and held-out evaluation."""
from __future__ import annotations

import json
from dataclasses import asdict
import logging
from pathlib import Path

from backend.app.worker.celery_app import celery
from backend.app.worker.env_artifacts import load_manifest
from backend.app.worker.env_lock import exclusive_environment
from backend.app.worker.progress import connect_progress_redis, progress_publisher, update_benchmark_run
from forge.settings import generated_envs_root

logger = logging.getLogger("backend.app.worker.tasks")


@celery.task(bind=True, name="backend.app.worker.tasks.run_benchmark_task")
def run_benchmark_task(
    self,
    run_id: str,
    domains: list[str],
    depth: int,
    seeds: int,
    output_dir: str,
) -> None:
    """Collect benchmark episodes and compute env quality metrics.

    Streams progress to Redis pub/sub channel forge:benchmark:{run_id}.
    Message types:
      {"total": N}           — total episodes to run
      {"log": "..."}         — human-readable log line
      {"progress": K}        — K episodes completed so far
      {"done": true}         — run complete
      {"error": "..."}       — run failed
    """
    try:
        r = connect_progress_redis()
    except Exception as exc:
        logger.error("[task:benchmark] Redis unavailable — %s", exc)
        update_benchmark_run(run_id, "failed", error=str(exc))
        return
    publish = progress_publisher(r, f"forge:benchmark:{run_id}", "benchmark")

    update_benchmark_run(run_id, "running")
    publish({"log": f"[benchmark] starting run {run_id} — domains={domains} depth={depth} seeds={seeds}"})

    try:
        from forge.benchmark.data_collector import DataCollector, CollectionConfig, CollectionCheckpoint
        from forge.benchmark.report import BenchmarkReport, ReportConfig

        from backend.app.database import get_session_factory
        from forge.benchmark.compiled_tasks import (
            CompiledTaskProvider,
            db_compiler_input_loader,
        )

        output_path = Path(output_dir)
        cfg = CollectionConfig(domains=domains, depth=depth, seeds=seeds, output_dir=output_path / "data")
        # Each selected environment is benchmarked against its own compiled tasks.
        task_provider = CompiledTaskProvider(
            loader=db_compiler_input_loader(get_session_factory())
        )
        collector = DataCollector(cfg, task_provider=task_provider)
        envs_root = generated_envs_root()

        for domain in domains:
            if not task_provider.tasks_for(domain=domain, depth=depth):
                publish({"log": f"  [skip] '{domain}' has no tasks at depth {depth} — compile/select an environment with tasks"})

        checkpoint = CollectionCheckpoint(output_dir=output_path / "data")
        pending = collector.pending_runs(checkpoint)
        total = len(pending)
        publish({"total": total})
        publish({"log": f"[benchmark] {total} episodes pending"})

        episodes = _BenchmarkEpisodes(envs_root=envs_root, redis_client=r, publish=publish)
        collector.collect(episodes)
        metrics = _env_quality_metrics(domains, output_path / "data", episodes.manifest_for, publish)

        report = BenchmarkReport(ReportConfig(output_dir=output_path))
        report.write_env_quality(metrics)

        report_data = [asdict(m) for m in metrics]
        update_benchmark_run(run_id, "done", report_json=json.dumps(report_data))
        publish({"done": True, "log": f"[benchmark] run complete — {len(metrics)} environments analyzed"})
        logger.info("[task:benchmark] run %s complete", run_id)

    except Exception as exc:
        logger.exception("[task:benchmark] run %s failed: %s", run_id, exc)
        update_benchmark_run(run_id, "failed", error=str(exc))
        publish({"error": str(exc)})

def _harbor_command(config: dict, executable: str) -> tuple[list[str], Path]:
    """Build a Harbor invocation without passing user input through a shell."""
    task_path = Path(config["harbor_task_path"]).resolve()
    agent = str(config["harbor_agent"])
    command = [executable, "run", "-p", task_path.name]
    if ":" in agent:
        command.extend(["--agent-import-path", agent])
    else:
        command.extend(["--agent", agent])
    command.extend(["--model", str(config["harbor_model"])])
    return command, task_path.parent

class _BenchmarkEpisodes:
    """Runs one benchmark episode per (task, seed) with a random agent, under the env lock."""

    def __init__(self, *, envs_root: Path, redis_client, publish) -> None:
        self._envs_root = envs_root
        self._redis = redis_client
        self._publish = publish
        self._manifests: dict[str, object] = {}
        self.completed = 0

    def manifest_for(self, domain: str):
        if domain not in self._manifests:
            self._manifests[domain] = load_manifest(self._envs_root / domain)
        return self._manifests[domain]

    def __call__(self, task, seed, jsonl_path) -> None:
        from backend.app.database import get_session_factory
        from forge.runtime.reliability import compute_environment_version, execute_reliable_episode

        port_file = self._envs_root / task.domain / "port"
        if not port_file.exists():
            self._publish({"log": f"  [skip] no port file for domain '{task.domain}' — start that environment first"})
            return
        env_version = compute_environment_version(env_name=task.domain, env_dir=self._envs_root / task.domain)
        with get_session_factory()() as db_bench:
            result, _ = execute_reliable_episode(
                task_runner=lambda attempt, attempt_seed: self._attempt(task, port_file, jsonl_path, attempt_seed),
                task_id=task.name,
                env_name=task.domain,
                environment_version=env_version,
                seed=seed,
                db_session=db_bench,
                logger=logger,
            )
        self.completed += 1
        self._publish({"log": f"  {task.name} seed={seed}  reward={result.total_reward:.3f}  reason={result.termination_reason}"})
        self._publish({"progress": self.completed})

    def _attempt(self, task, port_file: Path, jsonl_path, attempt_seed: int):
        from forge.envgen.agents.container_agent import make_container_agent
        from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig
        from forge.settings import experiment_seed

        port = int(port_file.read_text().strip())
        cfg_ep = EpisodeConfig(base_url=f"http://localhost:{port}", objective=task.objective)
        agent = make_container_agent("random", seed=experiment_seed(attempt_seed))
        with exclusive_environment(self._redis, task.domain), \
                ContainerEpisodeRunner(cfg_ep, manifest=self.manifest_for(task.domain)) as runner:
            return runner.run_episode(agent, jsonl_path=jsonl_path, seed=attempt_seed)


def _env_quality_metrics(domains: list[str], data_dir: Path, manifest_for, publish) -> list:
    """Env quality per domain that has a state manifest to measure coverage against."""
    from forge.benchmark.env_quality import compute_env_quality

    metrics = []
    for domain in domains:
        manifest = manifest_for(domain)
        if manifest:
            m = compute_env_quality(episode_dir=data_dir / domain, manifest=manifest)
            metrics.append(m)
            publish({"log": f"  {domain}: coverage={m.state_coverage_score:.2f}  dead_end_rate={m.dead_end_rate:.2f}"})
    return metrics


@celery.task(name="backend.app.worker.tasks.run_evaluation_task", ignore_result=True)
def run_evaluation_task(run_id: str, engine: str, config: dict) -> None:
    """Run a held-out Forge eval or an optional Harbor job with streamed logs."""
    try:
        redis_client = connect_progress_redis()
    except Exception as exc:
        logger.error("[task:eval] Redis unavailable — %s", exc)
        update_benchmark_run(run_id, "failed", error=str(exc))
        return
    publish = progress_publisher(redis_client, f"forge:benchmark:{run_id}", "eval")

    update_benchmark_run(run_id, "running")
    publish({"log": f"[eval] starting {engine} evaluation {run_id}"})
    try:
        if engine == "forge":
            from forge.benchmark._eval import evaluate_on_suite

            result = evaluate_on_suite(
                model_path=config["checkpoint"],
                suite=config["experiment"],
                seed=config.get("seed"),
                runs_dir=config["runs_dir"],
                run_id=run_id,
            )
            publish({
                "log": (
                    "[eval] held-out pass rate "
                    f"{result['heldout_pass_rate']:.3f}"
                )
            })
        elif engine == "harbor":
            import shutil
            import subprocess

            bundled = Path.cwd() / "example_tasks" / ".venv" / "bin" / "harbor"
            executable = shutil.which("harbor")
            if executable is None and bundled.is_file():
                executable = str(bundled)
            if executable is None:
                raise RuntimeError(
                    "Harbor is not installed. Run `uv tool install harbor` first."
                )
            command, working_dir = _harbor_command(config, executable)
            publish({"log": f"[eval] Harbor task: {working_dir / command[3]}"})
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    publish({"log": line.rstrip()})
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Harbor exited with status {return_code}")
            result = {
                "engine": "harbor",
                "status": "completed",
                "result_path": str(working_dir / "jobs"),
                "task_path": config["harbor_task_path"],
                "agent": config["harbor_agent"],
                "model": config["harbor_model"],
            }
        else:
            raise ValueError(f"unsupported evaluation engine: {engine}")

        update_benchmark_run(run_id, "done", report_json=json.dumps(result))
        publish({"done": True, "result": result, "log": "[eval] evaluation complete"})
    except Exception as exc:
        logger.exception("[task:eval] run %s failed: %s", run_id, exc)
        update_benchmark_run(run_id, "failed", error=str(exc))
        publish({"error": str(exc)})
