"""`forge benchmark ...`: collect, transfer, report and evaluate."""
from __future__ import annotations

import os
from pathlib import Path

import typer

# ── Benchmark sub-app ──────────────────────────────────────────────────────

benchmark_app = typer.Typer(name="benchmark", help="Benchmark Forge environments against established RL benchmarks.", no_args_is_help=True)


@benchmark_app.command("run")
def benchmark_run(
    domains: str = typer.Option(..., "--domains", help="Comma-separated generated environment names"),
    depth: int = typer.Option(5, "--depth", help="Max task depth (1–5)"),
    seeds: int = typer.Option(5, "--seeds", help="Number of seeds per task"),
    output: Path = typer.Option(Path("benchmark_results"), "--output", "-o"),
) -> None:
    """Collect episodes for the selected environments and compute env quality metrics."""
    from forge.benchmark.data_collector import DataCollector, CollectionConfig
    from forge.benchmark.compiled_tasks import CompiledTaskProvider, db_compiler_input_loader
    from forge.benchmark.env_quality import compute_env_quality
    from forge.benchmark.report import BenchmarkReport, ReportConfig
    from backend.app.database import get_session_factory

    domain_list = [d.strip() for d in domains.split(",") if d.strip()]
    cfg = CollectionConfig(domains=domain_list, depth=depth, seeds=seeds, output_dir=output / "data")
    provider = CompiledTaskProvider(loader=db_compiler_input_loader(get_session_factory()))
    collector = DataCollector(cfg, task_provider=provider)
    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))

    def run_episode(task, seed, jsonl_path):
        from forge.schema.state_schema import StateSchemaManifest
        from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig
        from forge.envgen.agents.container_agent import make_container_agent
        from forge.settings import experiment_seed

        manifest = None
        manifest_path = envs_root / task.domain / "state_schema.json"
        if manifest_path.exists():
            manifest = StateSchemaManifest.model_validate_json(manifest_path.read_text())

        port_file = envs_root / task.domain / "port"
        if not port_file.exists():
            typer.echo(f"  [skip] no port file for domain '{task.domain}' — run 'forge build' first", err=True)
            return
        port = int(port_file.read_text().strip())
        cfg_ep = EpisodeConfig(base_url=f"http://localhost:{port}", objective=task.objective)
        agent = make_container_agent("random", seed=experiment_seed(seed))
        with ContainerEpisodeRunner(cfg_ep, manifest=manifest) as runner:
            result = runner.run_episode(agent, jsonl_path=jsonl_path, seed=seed)
        typer.echo(f"  {task.name} seed={seed}  reward={result.total_reward:.3f}  reason={result.termination_reason}")

    typer.echo(f"[benchmark] collecting episodes (domains={domain_list} depth={depth} seeds={seeds})…")
    collector.collect(run_episode)

    metrics = []
    for domain in domain_list:
        manifest = None
        manifest_path = envs_root / domain / "state_schema.json"
        if manifest_path.exists():
            from forge.schema.state_schema import StateSchemaManifest
            manifest = StateSchemaManifest.model_validate_json(manifest_path.read_text())
        if manifest:
            m = compute_env_quality(episode_dir=output / "data" / domain, manifest=manifest)
            metrics.append(m)
            typer.echo(f"  {domain}: coverage={m.state_coverage_score:.2f}  dead_end_rate={m.dead_end_rate:.2f}")

    report = BenchmarkReport(ReportConfig(output_dir=output))
    report.write_env_quality(metrics)
    typer.echo(f"✓ Results written to {output}/")


@benchmark_app.command("transfer")
def benchmark_transfer(
    data: Path = typer.Option(..., "--data", help="Path to benchmark_results/data"),
    base_model: str = typer.Option("meta-llama/Llama-3.1-8B", "--base-model"),
    output: Path = typer.Option(Path("benchmark_results"), "--output", "-o"),
) -> None:
    """Reserved for a future external transfer benchmark. Requires GPU."""
    from forge.benchmark.transfer_pipeline import TransferConfig, run_transfer_pipeline
    cfg = TransferConfig(data_dir=data, base_model=base_model, output_dir=output)
    typer.echo(f"[benchmark] fine-tuning {base_model} on {data}…")
    try:
        result = run_transfer_pipeline(cfg)
    except NotImplementedError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"✓ Eval on {result.eval_suite}:")
    typer.echo(f"   task_completion_rate = {result.task_completion_rate:.3f}")
    typer.echo(f"   success@1 = {result.success_at_1:.3f}")
    typer.echo(f"   success@3 = {result.success_at_3:.3f}")


@benchmark_app.command("report")
def benchmark_report(
    output: Path = typer.Option(Path("benchmark_results"), "--output", "-o", help="Directory with collected results"),
) -> None:
    """Generate paper-ready figures and tables from collected results."""
    import json as _json
    from forge.benchmark.report import BenchmarkReport, ReportConfig
    from forge.benchmark.env_quality import EnvQualityMetrics

    summary_path = output / "summary.json"
    if not summary_path.exists():
        typer.echo(f"Error: {summary_path} not found. Run 'forge benchmark run' first.", err=True)
        raise typer.Exit(1)

    data = _json.loads(summary_path.read_text())
    metrics = [
        EnvQualityMetrics(
            env_name=m["env_name"],
            state_coverage_score=m["state_coverage_score"],
            reward_density=m["reward_density"],
            dead_end_rate=m["dead_end_rate"],
            action_diversity=m["action_diversity"],
            num_episodes=m["num_episodes"],
            num_steps=m["num_steps"],
        )
        for m in data.get("env_quality", [])
    ]
    report = BenchmarkReport(ReportConfig(output_dir=output))
    report.write_env_quality(metrics)
    typer.echo(f"✓ Figures written to {output}/figures/")


@benchmark_app.command("eval")
def benchmark_eval(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Directory containing policy_checkpoint.json"),
    experiment: Path = typer.Option(..., "--experiment", "--suite", help="Declarative experiment YAML"),
    seed: int | None = typer.Option(None, "--seed", help="Override checkpoint seed"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Result record root"),
) -> None:
    """Evaluate a policy checkpoint on internal held-out environments."""
    typer.echo(f"[benchmark] evaluating {checkpoint} on held-out split from {experiment}…")
    try:
        from forge.benchmark._eval import evaluate_on_suite
        result = evaluate_on_suite(
            model_path=str(checkpoint), suite=str(experiment), seed=seed,
            runs_dir=runs_dir,
        )
        typer.echo(f"✓ heldout_pass_rate = {result['heldout_pass_rate']:.3f}")
        typer.echo(f"  reward_hacking_rate = {result['reward_hacking_rate']:.3f}")
        typer.echo(f"  reward_variance = {result['reward_variance']:.3f}")
        typer.echo(f"  result = {result['result_path']}")
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
