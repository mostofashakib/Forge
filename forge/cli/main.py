from __future__ import annotations
import json
import os
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


def _verify_determinism(env_instance, seed: int) -> None:
    """Abort launch if two identically-seeded rollouts produce different observations."""
    from forge.runtime.determinism import DeterminismError, run_determinism_check
    try:
        report = run_determinism_check(env_instance, seed=seed)
    except DeterminismError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    if not report.skipped:
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
    policy = seeded_random_policy(seed)
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
    action_types = env_instance.action_types
    policy = seeded_random_policy(seed)
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


def _render_gym_replay(ep, steps, output_json: bool) -> None:
    """Render a compiled-gym episode (ep_* IDs, SQLite EpisodeStep rows)."""
    if output_json:
        out = {
            "episode_id": ep.id,
            "env_name": ep.env_name,
            "task_name": ep.task_name,
            "agent_id": ep.agent_id,
            "seed": ep.seed,
            "status": ep.status,
            "total_steps": ep.total_steps,
            "total_reward": ep.total_reward,
            "passed": ep.passed,
            "steps": [
                {
                    "step_index": s.step_index,
                    "action": json.loads(s.action),
                    "reward": s.reward,
                    "verifier_results": json.loads(s.verifier_results),
                    "diff": json.loads(s.diff),
                    "events": json.loads(s.events),
                    "state_hash_before": s.state_hash_before,
                    "state_hash_after": s.state_hash_after,
                    "terminated": s.terminated,
                    "truncated": s.truncated,
                }
                for s in steps
            ],
        }
        typer.echo(json.dumps(out, indent=2))
        return

    status_sym = "✓" if ep.passed else "✗"
    typer.echo(f"\nEpisode  {ep.id}")
    typer.echo(f"Env      {ep.env_name}  task={ep.task_name}  agent={ep.agent_id}  seed={ep.seed}")
    typer.echo(f"Result   {status_sym} {ep.status}  reward={ep.total_reward:+.3f}  steps={ep.total_steps}")
    typer.echo("")

    cumulative = 0.0
    for s in steps:
        cumulative += s.reward
        action = json.loads(s.action)
        act_str = action.get("type", "?")
        params = {k: v for k, v in action.items() if k != "type"}
        if params:
            act_str += "  " + "  ".join(f"{k}={v!r}" for k, v in params.items())

        reward_sym = "+" if s.reward > 0 else ("=" if s.reward == 0 else "-")
        term_note = "  [DONE]" if s.terminated else ("  [TRUNC]" if s.truncated else "")
        typer.echo(f"  {s.step_index:02d}  {reward_sym}  {s.reward:+.3f}  Σ{cumulative:+.3f}  {act_str}{term_note}")

        diff = json.loads(s.diff)
        if diff:
            for field, (before, after) in diff.items():
                before_str = json.dumps(before)
                after_str = json.dumps(after)
                if len(before_str) > 60:
                    before_str = before_str[:57] + "..."
                if len(after_str) > 60:
                    after_str = after_str[:57] + "..."
                typer.echo(f"        Δ {field}: {before_str} → {after_str}")

        try:
            vresults = json.loads(s.verifier_results)
        except (json.JSONDecodeError, TypeError):
            vresults = []
        for vr in vresults:
            if isinstance(vr, dict):
                vid = vr.get("verifier_id", "?")
                passed_v = vr.get("passed", False)
                score = vr.get("score", 0.0)
                vsym = "✓" if passed_v else "·"
                typer.echo(f"        {vsym} [{vid}] score={score:.2f}")

        try:
            events = json.loads(s.events)
        except (json.JSONDecodeError, TypeError):
            events = []
        for ev in events:
            if isinstance(ev, dict) and ev.get("type") == "policy_violation":
                rule = ev.get("rule_id", "?")
                typer.echo(f"        ⚠ policy violation: {rule}")

    typer.echo("")


def _render_container_replay(ep, step_records: list[dict], output_json: bool) -> None:
    """Render a containerized episode (cep_* IDs, JSONL step records)."""
    if output_json:
        out = {
            "episode_id": ep.id,
            "run_id": ep.run_id,
            "seed": ep.seed,
            "status": ep.status,
            "total_steps": ep.total_steps,
            "total_reward": ep.total_reward,
            "final_objective_score": ep.final_objective_score,
            "termination_reason": ep.termination_reason,
            "steps": step_records,
        }
        typer.echo(json.dumps(out, indent=2))
        return

    status_sym = "✓" if ep.status == "completed" and ep.total_reward > 0.5 else "✗"
    typer.echo(f"\nEpisode  {ep.id}  (container)")
    typer.echo(f"Run      {ep.run_id}  seed={ep.seed}")
    typer.echo(
        f"Result   {status_sym} {ep.status}  reward={ep.total_reward:+.3f}"
        f"  obj={ep.final_objective_score:.2f}  reason={ep.termination_reason or '?'}"
    )
    typer.echo("")

    cumulative = 0.0
    prev_hash: str | None = None
    for s in step_records:
        step_idx = s.get("step_index", "?")
        reward = s.get("reward", 0.0)
        obj_score = s.get("objective_score", None)
        hash_before = s.get("state_hash_before", "")
        hash_after = s.get("state_hash_after", "")
        terminated = s.get("terminated", False)
        truncated = s.get("truncated", False)
        term_reason = s.get("termination_reason")
        action = s.get("action", {})

        cumulative += reward
        endpoint = action.get("endpoint", action.get("type", "?"))
        payload = {k: v for k, v in action.items() if k not in ("endpoint", "type", "reasoning")}
        act_str = endpoint
        if payload:
            act_str += "  " + "  ".join(f"{k}={json.dumps(v)[:30]}" for k, v in list(payload.items())[:3])

        reward_sym = "+" if reward > 0 else ("=" if reward == 0 else "-")
        term_note = ""
        if term_reason:
            term_note = f"  [{term_reason}]"
        elif terminated:
            term_note = "  [DONE]"
        elif truncated:
            term_note = "  [TRUNC]"

        obj_str = f"  obj={obj_score:.2f}" if obj_score is not None else ""
        typer.echo(
            f"  {step_idx:02d}  {reward_sym}  {reward:+.3f}  Σ{cumulative:+.3f}{obj_str}  {act_str}{term_note}"
        )

        # Flag if state didn't change (dead-end indicator)
        if hash_before and hash_after and hash_before == hash_after:
            typer.echo("        ⚠ state unchanged (action had no effect)")
        elif prev_hash and hash_before != prev_hash:
            # State changed between steps without an action (non-determinism)
            typer.echo(f"        ⚠ state drift between steps ({prev_hash[:8]}→{hash_before[:8]})")

        prev_hash = hash_after

        # Show reasoning if present
        reasoning = action.get("reasoning", "")
        if reasoning and len(reasoning) > 0:
            short = reasoning[:120] + ("…" if len(reasoning) > 120 else "")
            typer.echo(f"        → {short}")

    typer.echo("")


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
                    for line in jsonl_path.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Skip the episode_summary line
                        if record.get("type") == "episode_summary":
                            continue
                        step_records.append(record)
                else:
                    typer.echo(f"Warning: JSONL file not found at {ep.jsonl_path}", err=True)
            else:
                typer.echo("Warning: no JSONL path recorded for this episode", err=True)

            _render_container_replay(ep, step_records, output_json)
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
        _render_gym_replay(ep, steps, output_json)


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
    url = db_url or os.environ.get("FORGE_DB_URL", "sqlite:///./forge.db")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)

    from backend.app.models import AgentEpisode, AgentRun, Episode

    # ── Load all episodes for this env ───────────────────────────────────
    with Session() as db:
        # Container episodes (cep_*)
        runs = db.query(AgentRun).filter_by(env_name=env_name).all()
        run_ids = [r.id for r in runs]
        container_eps: list[AgentEpisode] = []
        if run_ids:
            container_eps = (
                db.query(AgentEpisode)
                .filter(AgentEpisode.run_id.in_(run_ids))
                .all()
            )

        # Gym episodes (ep_*)
        gym_eps: list[Episode] = (
            db.query(Episode).filter_by(env_name=env_name).all()
        )

    total_container = len(container_eps)
    total_gym = len(gym_eps)

    if total_container == 0 and total_gym == 0:
        typer.echo(f"No episodes found for environment {env_name!r}.", err=True)
        raise typer.Exit(1)

    issues: list[str] = []
    recommendations: list[str] = []

    # ── Analyse container episodes ────────────────────────────────────────
    container_stats: dict = {}
    if total_container > 0:
        rewards = [e.total_reward for e in container_eps]
        obj_scores = [e.final_objective_score for e in container_eps]
        steps_list = [e.total_steps for e in container_eps]
        term_reasons: dict[str, int] = {}
        for e in container_eps:
            r = e.termination_reason or "unknown"
            term_reasons[r] = term_reasons.get(r, 0) + 1

        avg_reward = sum(rewards) / len(rewards)
        avg_obj = sum(obj_scores) / len(obj_scores)
        avg_steps = sum(steps_list) / len(steps_list)
        pass_rate = sum(1 for r in rewards if r > 0.5) / len(rewards)

        # Per-step analysis from JSONL files
        dead_end_steps = 0
        total_steps_scanned = 0
        reward_zero_streak_episodes = 0
        state_drift_episodes = 0
        action_counts: dict[str, int] = {}

        for ep in container_eps:
            if not ep.jsonl_path:
                continue
            jsonl_path = Path(ep.jsonl_path)
            if not jsonl_path.exists():
                continue

            step_records = []
            for line in jsonl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "episode_summary":
                    continue
                step_records.append(rec)

            total_steps_scanned += len(step_records)

            # Dead-end: action didn't change state
            for s in step_records:
                hb = s.get("state_hash_before", "")
                ha = s.get("state_hash_after", "")
                if hb and ha and hb == ha:
                    dead_end_steps += 1
                endpoint = s.get("action", {}).get("endpoint", s.get("action", {}).get("type", ""))
                if endpoint:
                    action_counts[endpoint] = action_counts.get(endpoint, 0) + 1

            # Reward zero streak: all steps after step 0 have reward == 0
            non_first_rewards = [s.get("reward", 0) for s in step_records[1:]]
            if non_first_rewards and all(r == 0.0 for r in non_first_rewards):
                reward_zero_streak_episodes += 1

            # State drift: state_hash_before of step N ≠ state_hash_after of step N-1
            for i in range(1, len(step_records)):
                prev_after = step_records[i - 1].get("state_hash_after", "")
                curr_before = step_records[i].get("state_hash_before", "")
                if prev_after and curr_before and prev_after != curr_before:
                    state_drift_episodes += 1
                    break

        dead_end_rate = dead_end_steps / max(total_steps_scanned, 1)

        container_stats = {
            "total_episodes": total_container,
            "avg_reward": round(avg_reward, 4),
            "avg_final_obj_score": round(avg_obj, 4),
            "avg_steps": round(avg_steps, 2),
            "pass_rate": round(pass_rate, 4),
            "termination_reasons": term_reasons,
            "dead_end_rate": round(dead_end_rate, 4),
            "reward_zero_streak_episodes": reward_zero_streak_episodes,
            "state_drift_episodes": state_drift_episodes,
            "top_actions": sorted(action_counts.items(), key=lambda x: -x[1])[:10],
        }

        # ── Issue detection ───────────────────────────────────────────────
        if avg_reward < 0.1:
            issues.append(f"Extremely low avg reward ({avg_reward:.3f}) — objective scorer may be miscalibrated or environment is unsolvable")
            recommendations.append("Check /forge/state output: does it expose the fields the objective needs to evaluate progress?")

        if dead_end_rate > 0.3:
            issues.append(f"High dead-end rate ({dead_end_rate:.0%} of steps leave state unchanged) — actions not affecting state")
            recommendations.append("Inspect which endpoints have unchanged state hashes and verify the app's state bridge includes their effects in /forge/state")

        diverge_count = term_reasons.get("diverged", 0)
        if diverge_count / total_container > 0.5:
            issues.append(f"{diverge_count}/{total_container} episodes terminated due to 'diverged' — scorer drops to 0.0 prematurely")
            recommendations.append("The objective scorer gives 0.0 after some actions even though state changed — the objective may be too vague or the state representation too large for the LLM scorer to parse correctly")

        if reward_zero_streak_episodes / max(total_container, 1) > 0.5:
            issues.append(f"{reward_zero_streak_episodes}/{total_container} episodes have reward=0 for all steps after step 0 — reward signal is nearly zero-shot")
            recommendations.append("Only the first action ever gets non-zero reward, suggesting the scorer immediately plateaus at 0.0 after the first state change — the state bridge may be returning a stale snapshot")

        if state_drift_episodes > 0:
            issues.append(f"State drift in {state_drift_episodes}/{total_container} episodes — state changes between steps without an action (non-determinism)")
            recommendations.append("Remove timestamps, UUIDs, or auto-incrementing IDs from /forge/state output; these make the reward signal noisy across seeds")

        if avg_steps < 5:
            issues.append(f"Episodes terminate very early (avg {avg_steps:.1f} steps) — environment may be too hard or divergence threshold too aggressive")
            recommendations.append("Consider reducing divergence_threshold (currently 0.2) or increasing max_steps for this environment type")

    # ── Analyse gym episodes ──────────────────────────────────────────────
    gym_stats: dict = {}
    if total_gym > 0:
        rewards_g = [e.total_reward for e in gym_eps]
        passes = [e.passed for e in gym_eps]
        gym_stats = {
            "total_episodes": total_gym,
            "avg_reward": round(sum(rewards_g) / len(rewards_g), 4),
            "pass_rate": round(sum(passes) / len(passes), 4),
        }
        if sum(passes) == 0:
            issues.append("No gym episodes passed — verifiers may be misconfigured or tasks unsolvable")
            recommendations.append("Check verifier expressions in the policy DSL; they may reference fields not present in the state")

    # ── Output ────────────────────────────────────────────────────────────
    if output_json:
        typer.echo(json.dumps({
            "env_name": env_name,
            "container_episodes": container_stats,
            "gym_episodes": gym_stats,
            "issues": issues,
            "recommendations": recommendations,
        }, indent=2))
        return

    typer.echo(f"\nDiagnosis: {env_name}\n{'─' * 50}")

    if container_stats:
        typer.echo(f"\nContainer episodes ({total_container})")
        typer.echo(f"  Pass rate:       {container_stats['pass_rate']:.0%}")
        typer.echo(f"  Avg reward:      {container_stats['avg_reward']:+.3f}")
        typer.echo(f"  Avg obj score:   {container_stats['avg_final_obj_score']:.3f}")
        typer.echo(f"  Avg steps:       {container_stats['avg_steps']:.1f}")
        typer.echo(f"  Dead-end rate:   {container_stats['dead_end_rate']:.0%}  (steps where action had no effect)")
        typer.echo(f"  Zero-streak eps: {container_stats['reward_zero_streak_episodes']}/{total_container}")
        typer.echo(f"  State drift eps: {container_stats['state_drift_episodes']}/{total_container}")
        typer.echo("\n  Termination reasons:")
        for reason, count in sorted(container_stats["termination_reasons"].items(), key=lambda x: -x[1]):
            typer.echo(f"    {count:3d}×  {reason}")
        if container_stats["top_actions"]:
            typer.echo("\n  Most-called endpoints:")
            for endpoint, count in container_stats["top_actions"][:5]:
                typer.echo(f"    {count:3d}×  {endpoint}")

    if gym_stats:
        typer.echo(f"\nGym episodes ({total_gym})")
        typer.echo(f"  Pass rate:  {gym_stats['pass_rate']:.0%}")
        typer.echo(f"  Avg reward: {gym_stats['avg_reward']:+.3f}")

    if issues:
        typer.echo(f"\nIssues found ({len(issues)})")
        for i, issue in enumerate(issues, 1):
            typer.echo(f"  {i}. {issue}")
        typer.echo("\nRecommendations")
        for i, rec in enumerate(recommendations, 1):
            typer.echo(f"  {i}. {rec}")
    else:
        typer.echo("\n✓ No quality issues detected")

    typer.echo("")


# ── Benchmark sub-app ──────────────────────────────────────────────────────

benchmark_app = typer.Typer(name="benchmark", help="Benchmark Forge environments against established RL benchmarks.", no_args_is_help=True)
app.add_typer(benchmark_app)


@benchmark_app.command("run")
def benchmark_run(
    domains: str = typer.Option("email,project_mgmt", "--domains", help="Comma-separated domain names"),
    depth: int = typer.Option(5, "--depth", help="Max task depth (1–5)"),
    seeds: int = typer.Option(5, "--seeds", help="Number of seeds per task"),
    output: Path = typer.Option(Path("benchmark_results"), "--output", "-o"),
) -> None:
    """Collect episodes across the task suite and compute env quality metrics."""
    from forge.benchmark.data_collector import DataCollector, CollectionConfig
    from forge.benchmark.env_quality import compute_env_quality
    from forge.benchmark.report import BenchmarkReport, ReportConfig

    domain_list = [d.strip() for d in domains.split(",")]
    cfg = CollectionConfig(domains=domain_list, depth=depth, seeds=seeds, output_dir=output / "data")
    collector = DataCollector(cfg)
    envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))

    def run_episode(task, seed, jsonl_path):
        from forge.schema.state_schema import StateSchemaManifest
        from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig
        from forge.envgen.agents.container_agent import make_container_agent

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
        agent = make_container_agent("random", seed=seed)
        with ContainerEpisodeRunner(cfg_ep, manifest=manifest) as runner:
            result = runner.run_episode(agent, jsonl_path=jsonl_path)
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
    """Fine-tune base model on Forge data and evaluate zero-shot on WebArena/WorkArena. Requires GPU."""
    from forge.benchmark.transfer_pipeline import TransferConfig, run_transfer_pipeline
    cfg = TransferConfig(data_dir=data, base_model=base_model, output_dir=output)
    typer.echo(f"[benchmark] fine-tuning {base_model} on {data}…")
    result = run_transfer_pipeline(cfg)
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
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to fine-tuned model checkpoint"),
    suite: str = typer.Option("webArena", "--suite", help="Eval suite: webArena or workArena"),
) -> None:
    """Evaluate a fine-tuned checkpoint zero-shot on WebArena or WorkArena."""
    typer.echo(f"[benchmark] evaluating {checkpoint} on {suite}…")
    try:
        from forge.benchmark._eval import evaluate_on_suite
        result = evaluate_on_suite(model_path=str(checkpoint), suite=suite)
        typer.echo(f"✓ task_completion_rate = {result['task_completion_rate']:.3f}")
    except ImportError:
        typer.echo("Error: evaluation requires 'transformers' and the eval harness. See docs/superpowers/specs/ for setup.", err=True)
        raise typer.Exit(1)
