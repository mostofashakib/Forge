from __future__ import annotations
import json
import sys
from pathlib import Path
import typer

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
    task_dict: dict | None = {"name": task, "verifier_id": task} if task else None
    obs, info = env_instance.reset(seed=seed, options={"task": task_dict} if task_dict else None)
    typer.echo(f"Episode: {info['episode_id']}  task={task or 'none'}  seed={seed}")

    action_types = sorted(env_instance._transition_engine.action_types)
    if not action_types:
        typer.echo("No actions registered.")
        return

    import random
    rng = random.Random(seed)
    for step_num in range(steps):
        action_type = rng.choice(action_types)
        action = {"type": action_type}
        obs, step_reward, terminated, truncated, step_info = env_instance.step(action)
        typer.echo(f"  step {step_num:02d}: {action_type:<30} reward={step_reward:+.3f}")
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
    task_dict: dict | None = {"name": task, "verifier_id": task} if task else None
    env_instance.reset(seed=seed, options={"task": task_dict} if task_dict else None)

    action_types = sorted(env_instance._transition_engine.action_types)
    import random
    rng = random.Random(seed)
    for _ in range(steps):
        if not action_types:
            break
        action = {"type": rng.choice(action_types)}
        _, _, terminated, truncated, _ = env_instance.step(action)
        if terminated or truncated:
            break

    out.mkdir(parents=True, exist_ok=True)
    episode_id = env_instance._episode_id or "ep_unknown"
    out_file = out / f"{episode_id}.jsonl"
    out_file.write_text(env_instance._traj_store.to_jsonl())
    typer.echo(f"✓ Exported → {out_file}")
