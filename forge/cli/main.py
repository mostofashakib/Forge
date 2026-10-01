from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import typer

from forge.cli import replay_render
from forge.contracts.episode import read_trajectory_steps

app = typer.Typer(
    name="forge",
    help="Forge RL Environment Platform — compile, validate, run, and export environments.",
    no_args_is_help=True,
)


@app.command()
def compile(
    input: Path = typer.Option(..., "--input", "-i", help="Path to CompilerInput JSON file"),
    output: Path = typer.Option(Path("generated_envs"), "--output", "-o", help="Output root directory"),
    validate: bool = typer.Option(True, help="Run validation after building"),
) -> None:
    """Compile a CompilerInput JSON into a generated environment package."""
    if not input.exists():
        typer.echo(f"Error: input file not found: {input}", err=True)
        raise typer.Exit(1)
    try:
        raw = input.read_text()
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        typer.echo(f"Error: invalid JSON — {exc}", err=True)
        raise typer.Exit(1)

    from forge.extraction.schemas import CompilerInput
    try:
        compiler_input = CompilerInput.model_validate(data)
    except Exception as exc:
        typer.echo(f"Error: invalid CompilerInput schema — {exc}", err=True)
        raise typer.Exit(1)

    from forge.compiler.package_builder import PackageBuilder
    output.mkdir(parents=True, exist_ok=True)
    pkg_dir = PackageBuilder(output).build(compiler_input)
    typer.echo(f"✓ Generated: {pkg_dir}")

    if validate:
        from forge.compiler.validation_runner import ValidationRunner
        result = ValidationRunner(output).run(pkg_dir)
        if result.total_tests > 0:
            status = "passed" if result.passed else "FAILED"
            typer.echo(f"  Tests: {result.total_tests} {status}")
        else:
            typer.echo("  Tests: no tests found (generated stubs may not yet pass)")

        from forge.compiler.override_validator import OverrideValidator
        ov_result = OverrideValidator().validate(pkg_dir, compiler_input)
        if not ov_result.valid:
            typer.echo("  Override validation FAILED:", err=True)
            for err in ov_result.errors:
                typer.echo(f"    - {err}", err=True)
            raise typer.Exit(1)
        typer.echo("  Overrides: valid")


@app.command()
def validate(
    env_dir: Path = typer.Argument(..., help="Path to generated environment directory"),
) -> None:
    """Run tests and override validation on a generated environment."""
    if not env_dir.exists():
        typer.echo(f"Error: directory not found: {env_dir}", err=True)
        raise typer.Exit(1)

    from forge.compiler.validation_runner import ValidationRunner
    root = env_dir.parent
    result = ValidationRunner(root).run(env_dir)

    if result.total_tests > 0:
        status = "passed" if result.passed else "FAILED"
        typer.echo(f"Tests: {result.total_tests} {status}")
        if not result.passed:
            typer.echo(result.output)
            raise typer.Exit(1)
    else:
        typer.echo("Tests: no tests found")

    typer.echo("✓ Valid")


def _verify_determinism(env_instance, seed: int) -> None:
    """Abort launch if two identically-seeded rollouts produce different observations."""
    from forge.runtime.determinism import DeterminismError, run_determinism_check
    from forge.settings import determinism_enabled
    if not determinism_enabled():
        typer.echo("Determinism check disabled (FORGE_DETERMINISM=off)")
        return
    try:
        report = run_determinism_check(env_instance, seed=seed)
    except DeterminismError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Determinism check passed (obs hash {report.observation_hash[:16]})")


@app.command()
def run(
    env: str = typer.Option(..., "--env", help="Environment name under generated_envs/"),
    task: str = typer.Option("", "--task", help="Task name (verifier_id). Omit to skip verification."),
    seed: int = typer.Option(42, "--seed"),
    steps: int = typer.Option(10, "--steps"),
    envs_dir: Path = typer.Option(Path("generated_envs"), "--envs-dir", hidden=True),
) -> None:
    """Run one episode of a generated environment and print step-by-step output."""
    root = Path.cwd()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import importlib
    try:
        mod = importlib.import_module(f"generated_envs.{env}.gym_wrapper")
    except ImportError as exc:
        typer.echo(f"Error: could not import environment '{env}': {exc}", err=True)
        raise typer.Exit(1)

    build_fn_name = f"build_{env}_env"
    if not hasattr(mod, build_fn_name):
        typer.echo(f"Error: {build_fn_name} not found in gym_wrapper", err=True)
        raise typer.Exit(1)

    env_instance = getattr(mod, build_fn_name)(max_steps=steps)
    _verify_determinism(env_instance, seed)
    task_dict: dict | None = {"name": task, "verifier_id": task} if task else None
    obs, info = env_instance.reset(seed=seed, options={"task": task_dict} if task_dict else None)
    typer.echo(f"Episode: {info['episode_id']}  task={task or 'none'}  seed={seed}")

    action_types = env_instance.action_types
    if not action_types:
        typer.echo("No actions registered.")
        return

    from forge.runtime.policy import seeded_random_policy
    from forge.settings import experiment_seed
    policy = seeded_random_policy(experiment_seed(seed))
    for step_num in range(steps):
        action = policy(obs, action_types)
        obs, step_reward, terminated, truncated, step_info = env_instance.step(action)
        typer.echo(f"  step {step_num:02d}: {action['type']:<30} reward={step_reward:+.3f}")
        if terminated:
            typer.echo("  → Terminated (task succeeded)")
            break
        if truncated:
            typer.echo("  → Truncated (max steps reached)")
            break

    typer.echo("Done.")


@app.command()
def export(
    env: str = typer.Option(..., "--env", help="Environment name"),
    task: str = typer.Option("", "--task", help="Task name"),
    seed: int = typer.Option(42, "--seed"),
    steps: int = typer.Option(20, "--steps"),
    out: Path = typer.Option(Path("exports"), "--out"),
) -> None:
    """Run an episode and export the trajectory as JSONL."""
    root = Path.cwd()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import importlib
    try:
        mod = importlib.import_module(f"generated_envs.{env}.gym_wrapper")
    except ImportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    build_fn_name = f"build_{env}_env"
    env_instance = getattr(mod, build_fn_name)(max_steps=steps)
    _verify_determinism(env_instance, seed)
    task_dict: dict | None = {"name": task, "verifier_id": task} if task else None
    env_instance.reset(seed=seed, options={"task": task_dict} if task_dict else None)

    from forge.runtime.policy import seeded_random_policy
    from forge.settings import experiment_seed
    action_types = env_instance.action_types
    policy = seeded_random_policy(experiment_seed(seed))
    for _ in range(steps):
        if not action_types:
            break
        _, _, terminated, truncated, _ = env_instance.step(policy(None, action_types))
        if terminated or truncated:
            break

    out.mkdir(parents=True, exist_ok=True)
    episode_id = env_instance._episode_id or "ep_unknown"
    out_file = out / f"{episode_id}.jsonl"
    out_file.write_text(env_instance._traj_store.to_jsonl())
    typer.echo(f"✓ Exported → {out_file}")


@app.command()
def replay(
    episode_id: str = typer.Argument(..., help="Episode ID to replay (ep_* for gym envs, cep_* for container envs)"),
    db_url: str = typer.Option("", "--db", help="SQLite DB URL (default: $FORGE_DB_URL or sqlite:///./forge.db)"),
    output_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Display a recorded episode step-by-step.

    Supports both compiled gym environments (ep_* IDs) and containerized
    environments (cep_* IDs). Container episodes read their step data from
    the JSONL file recorded alongside the SQLite metadata.
    """
    url = db_url or os.environ.get("FORGE_DB_URL", "sqlite:///./forge.db")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)

    # ── Container episode (cep_* prefix) ─────────────────────────────────
    if episode_id.startswith("cep_"):
        from backend.app.models import AgentEpisode
        with Session() as db:
            ep: AgentEpisode | None = db.get(AgentEpisode, episode_id)
            if ep is None:
                typer.echo(f"Error: episode {episode_id!r} not found in {url}", err=True)
                raise typer.Exit(1)

            step_records: list[dict] = []
            if ep.jsonl_path:
                jsonl_path = Path(ep.jsonl_path)
                if jsonl_path.exists():
                    step_records = list(read_trajectory_steps(jsonl_path))
                else:
                    typer.echo(f"Warning: JSONL file not found at {ep.jsonl_path}", err=True)
            else:
                typer.echo("Warning: no JSONL path recorded for this episode", err=True)

            replay_render.render_container_replay(ep, step_records, output_json)
        return

    # ── Compiled gym episode (ep_* prefix) ───────────────────────────────
    from backend.app.models import Episode, EpisodeStep
    with Session() as db:
        ep: Episode | None = db.get(Episode, episode_id)
        if ep is None:
            typer.echo(f"Error: episode {episode_id!r} not found in {url}", err=True)
            raise typer.Exit(1)

        steps = (
            db.query(EpisodeStep)
            .filter_by(episode_id=episode_id)
            .order_by(EpisodeStep.step_index)
            .all()
        )
        replay_render.render_gym_replay(ep, steps, output_json)


@app.command()
def diagnose(
    env_name: str = typer.Argument(..., help="Environment name to diagnose"),
    db_url: str = typer.Option("", "--db", help="SQLite DB URL"),
    envs_dir: Path = typer.Option(Path("generated_envs"), "--envs-dir", help="Generated envs root"),
    output_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Analyze episode quality across all runs for an environment.

    Surfaces systemic problems that replay misses: dead-end patterns,
    reward sparsity, scorer inconsistency, non-deterministic state drift,
    and termination-reason distributions.
    """
    from forge.cli import diagnosis

    url = db_url or os.environ.get("FORGE_DB_URL", "sqlite:///./forge.db")
    container_eps, gym_eps = diagnosis.load_episodes(url, env_name)
    if not container_eps and not gym_eps:
        typer.echo(f"No episodes found for environment {env_name!r}.", err=True)
        raise typer.Exit(1)

    result = diagnosis.diagnose(env_name, container_eps, gym_eps)
    if output_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return
    diagnosis.render(result)


@app.command()
def train(
    data_dir: Path = typer.Option(..., "--data", help="Directory containing exported graded rollouts"),
    base_model: str | None = typer.Option(None, "--base-model", help="Base policy checkpoint (or read from --experiment)"),
    output: Path = typer.Option(Path("forge_policy"), "--output", "-o", help="Checkpoint output directory"),
    objective: str = typer.Option("grpo", "--objective", help="Training objective: grpo | dpo | sft | ppo"),
    max_steps: int = typer.Option(500, "--max-steps"),
    experiment: Path | None = typer.Option(None, "--experiment", help="Declarative experiment YAML"),
    seed: int | None = typer.Option(None, "--seed", help="Experiment seed; required when YAML lists more than one"),
) -> None:
    """Train Forge's own policy from its own graded rollouts (GRPO / DPO / SFT / PPO).

    Consumes `grpo_rollouts.parquet` (GRPO) or `preference_pairs.jsonl` (DPO)
    exported from graded episodes and writes a policy checkpoint the runtime
    agents can load. Requires trl + transformers on a GPU node.
    """
    from forge.training.trainer import (
        NoTrainingSignalError,
        PolicyTrainer,
        TrainingConfig,
        TrainingObjective,
    )
    from forge.training.dataset import MalformedExportError

    experiment_config = None
    train_envs = None
    run_id = ""
    if experiment is not None:
        from datetime import datetime, timezone
        from forge.experiments import ExperimentConfig

        try:
            experiment_config = ExperimentConfig.load(experiment)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"Invalid experiment: {exc}", err=True)
            raise typer.Exit(2)
        if seed is None:
            if len(experiment_config.seeds) != 1:
                typer.echo("Error: --seed is required when the experiment lists multiple seeds", err=True)
                raise typer.Exit(2)
            seed = experiment_config.seeds[0]
        if seed not in experiment_config.seeds:
            typer.echo(f"Error: seed {seed} is not declared by {experiment}", err=True)
            raise typer.Exit(2)
        if base_model is not None and base_model != experiment_config.base_model:
            typer.echo("Error: --base-model conflicts with the experiment config", err=True)
            raise typer.Exit(2)
        base_model = experiment_config.base_model
        train_envs = experiment_config.train_envs
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{experiment.stem}-s{seed}-{timestamp}"
    elif base_model is None:
        typer.echo("Error: provide --base-model or --experiment", err=True)
        raise typer.Exit(2)

    try:
        obj = TrainingObjective(objective.lower())
    except ValueError:
        valid = ", ".join(repr(o.value) for o in TrainingObjective)
        typer.echo(f"Error: unknown objective {objective!r}; use one of {valid}", err=True)
        raise typer.Exit(2)

    try:
        result = PolicyTrainer().train(TrainingConfig(
            data_dir=data_dir, base_model=base_model, output_dir=output,
            objective=obj, max_steps=max_steps, train_envs=train_envs,
            experiment_config=(experiment_config.model_dump(mode="json") if experiment_config else None),
            seed=seed, run_id=run_id,
        ))
    except NoTrainingSignalError as exc:
        typer.echo(f"No training signal: {exc}", err=True)
        raise typer.Exit(1)
    except MalformedExportError as exc:
        typer.echo(f"Malformed export: {exc}", err=True)
        raise typer.Exit(1)
    except (RuntimeError, NotImplementedError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"Trained {result.objective} policy on {result.num_examples} examples "
        f"(mean reward {result.mean_reward:.3f}) → {result.checkpoint_path}"
    )
    if run_id:
        typer.echo(f"Run ID: {run_id}")


from forge.cli.benchmark import benchmark_app  # noqa: E402  (registered after the core commands)

app.add_typer(benchmark_app)
