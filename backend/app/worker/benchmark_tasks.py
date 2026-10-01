"""Celery tasks for benchmark collection, held-out evaluation and transfer evaluation."""
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
    from functools import cache

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
        from forge.benchmark.env_quality import compute_env_quality
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

        @cache
        def manifest_for(domain: str):
            return load_manifest(envs_root / domain)

        for domain in domains:
            if not task_provider.tasks_for(domain=domain, depth=depth):
                publish({"log": f"  [skip] '{domain}' has no tasks at depth {depth} — compile/select an environment with tasks"})

        checkpoint = CollectionCheckpoint(output_dir=output_path / "data")
        pending = collector.pending_runs(checkpoint)
        total = len(pending)
        publish({"total": total})
        publish({"log": f"[benchmark] {total} episodes pending"})

        completed = 0

        def run_episode(task, seed, jsonl_path):
            nonlocal completed
            from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig
            from forge.envgen.agents.container_agent import make_container_agent
            from forge.settings import experiment_seed
            from forge.runtime.reliability import execute_reliable_episode, compute_environment_version

            manifest = manifest_for(task.domain)
            port_file = envs_root / task.domain / "port"
            if not port_file.exists():
                publish({"log": f"  [skip] no port file for domain '{task.domain}' — start that environment first"})
                return

            env_version = compute_environment_version(env_name=task.domain, env_dir=envs_root / task.domain)

            def _benchmark_attempt(attempt: int, attempt_seed: int):
                port = int(port_file.read_text().strip())
                cfg_ep = EpisodeConfig(base_url=f"http://localhost:{port}", objective=task.objective)
                agent = make_container_agent("random", seed=experiment_seed(attempt_seed))
                with exclusive_environment(r, task.domain), \
                        ContainerEpisodeRunner(cfg_ep, manifest=manifest) as runner:
                    return runner.run_episode(agent, jsonl_path=jsonl_path, seed=attempt_seed)

            with get_session_factory()() as db_bench:
                result, _ = execute_reliable_episode(
                    task_runner=_benchmark_attempt,
                    task_id=task.name,
                    env_name=task.domain,
                    environment_version=env_version,
                    seed=seed,
                    db_session=db_bench,
                    logger=logger,
                )

            completed += 1
            publish({"log": f"  {task.name} seed={seed}  reward={result.total_reward:.3f}  reason={result.termination_reason}"})
            publish({"progress": completed})

        collector.collect(run_episode)

        metrics = []
        for domain in domains:
            manifest = manifest_for(domain)
            if manifest:
                m = compute_env_quality(episode_dir=output_path / "data" / domain, manifest=manifest)
                metrics.append(m)
                publish({"log": f"  {domain}: coverage={m.state_coverage_score:.2f}  dead_end_rate={m.dead_end_rate:.2f}"})

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

@celery.task(name="backend.app.worker.tasks.run_transfer_task", ignore_result=True)
def run_transfer_task(run_id: str, config: dict) -> None:
    """Run a transfer evaluation with the production pipeline and stream its progress."""
    from forge.benchmark.transfer_pipeline import run_transfer_pipeline

    redis_client = None
    try:
        redis_client = connect_progress_redis()
    except Exception as exc:
        logger.warning("[task:transfer] Redis unavailable — %s", exc)
    execute_transfer_run(
        run_id,
        config,
        evaluate=run_transfer_pipeline,
        publish=progress_publisher(redis_client, f"forge:benchmark:{run_id}", "transfer"),
    )


def execute_transfer_run(run_id: str, config: dict, *, evaluate, publish) -> None:
    """Run `evaluate` on the request and record exactly what it measured.

    `evaluate` is a `TransferEvaluator`. A failing evaluator fails the run, so
    no metric is ever reported that an evaluation did not produce.
    """
    from forge.benchmark.transfer_pipeline import TransferConfig

    update_benchmark_run(run_id, "running")
    transfer_config = TransferConfig(
        data_dir=Path(config.get("data_dir", "benchmark_results/data")),
        base_model=config.get("base_model", "meta-llama/Llama-3.1-8B"),
        output_dir=Path(config.get("output_dir", "benchmark_results/transfer")),
        eval_suite=config.get("eval_suite", "held-out-transfer"),
        max_train_steps=config.get("max_steps", 500),
        seeds=config.get("seeds", 3),
    )
    inference_mode = config.get("inference_mode", "auto")
    device = _detected_device()
    publish({"log": f"[transfer] starting transfer benchmark {run_id}"})
    publish({"log": f"[transfer] base model: {transfer_config.base_model}"})
    publish({"log": f"[transfer] transfer dataset: {transfer_config.data_dir}"})
    publish({"log": f"[transfer] target suite: {transfer_config.eval_suite}"})
    publish({"log": f"[transfer] inference configuration: {inference_mode}"})
    publish({"log": f"[transfer] hardware detected: {device}"})

    try:
        measured = evaluate(transfer_config)
    except Exception as exc:
        logger.exception("[task:transfer] run %s failed: %s", run_id, exc)
        update_benchmark_run(run_id, "failed", error=str(exc))
        publish({"error": str(exc)})
        return

    result = {**measured.to_dict(), "inference_mode": inference_mode, "device": device}
    publish({"log": f"[transfer] evaluated {measured.num_eval_tasks} tasks across {transfer_config.seeds} seeds"})
    publish({"log": f"[transfer] task completion rate: {measured.task_completion_rate * 100:.1f}%"})
    publish({"log": f"[transfer] pass@1: {measured.success_at_1 * 100:.1f}% | pass@3: {measured.success_at_3 * 100:.1f}%"})
    publish({"done": True, "result": result})
    update_benchmark_run(run_id, "done", report_json=json.dumps(result))


def _detected_device() -> str:
    from forge.contracts.gpu import GPUDeviceSpec

    hardware = GPUDeviceSpec.probe_hardware()
    if hardware.get("cuda_available"):
        devices = hardware.get("devices") or [{}]
        return f"CUDA GPU ({devices[0].get('name', 'NVIDIA')})"
    if hardware.get("mps_available"):
        return "Apple Silicon MPS (Metal)"
    return "CPU (standard)"
