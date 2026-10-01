"""Print a recorded episode step by step, or as JSON."""
from __future__ import annotations

import json

import typer


def render_gym_replay(ep, steps, output_json: bool) -> None:
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


def render_container_replay(ep, step_records: list[dict], output_json: bool) -> None:
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
