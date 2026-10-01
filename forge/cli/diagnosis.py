"""Find systemic quality problems across every recorded episode of one environment.

Loading, statistics, issue detection and rendering are separate steps, so
the statistics and the rules that read them stay pure and testable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import typer

from forge.contracts.episode import read_trajectory_steps


@dataclass
class Diagnosis:
    env_name: str
    container: dict = field(default_factory=dict)
    gym: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def flag(self, issue: str, recommendation: str) -> None:
        self.issues.append(issue)
        self.recommendations.append(recommendation)

    def to_dict(self) -> dict:
        return {
            "env_name": self.env_name,
            "container_episodes": self.container,
            "gym_episodes": self.gym,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


def load_episodes(db_url: str, env_name: str) -> tuple[list, list]:
    """(container episodes, gym episodes) recorded for `env_name`."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.app.models import AgentEpisode, AgentRun, Episode

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    with sessionmaker(bind=engine)() as db:
        container_eps = (
            db.query(AgentEpisode)
            .join(AgentRun, AgentEpisode.run_id == AgentRun.id)
            .filter(AgentRun.env_name == env_name)
            .all()
        )
        gym_eps = db.query(Episode).filter_by(env_name=env_name).all()
    return container_eps, gym_eps


def diagnose(env_name: str, container_eps: list, gym_eps: list) -> Diagnosis:
    diagnosis = Diagnosis(env_name)
    if container_eps:
        stats = container_stats(container_eps)
        diagnosis.container = _rounded(stats)
        _flag_container_issues(diagnosis, stats)
    if gym_eps:
        passes = [e.passed for e in gym_eps]
        diagnosis.gym = {
            "total_episodes": len(gym_eps),
            "avg_reward": round(sum(e.total_reward for e in gym_eps) / len(gym_eps), 4),
            "pass_rate": round(sum(passes) / len(passes), 4),
        }
        if sum(passes) == 0:
            diagnosis.flag(
                "No gym episodes passed — verifiers may be misconfigured or tasks unsolvable",
                "Check verifier expressions in the policy DSL; they may reference fields not present in the state",
            )
    return diagnosis


def container_stats(episodes: list) -> dict:
    total = len(episodes)
    rewards = [e.total_reward for e in episodes]
    avg_steps = sum(e.total_steps for e in episodes) / total
    term_reasons = Counter(e.termination_reason or "unknown" for e in episodes)

    dead_end_steps = scanned_steps = zero_streak = drift = 0
    action_counts: Counter[str] = Counter()
    for episode in episodes:
        if not episode.jsonl_path or not Path(episode.jsonl_path).exists():
            continue
        records = list(read_trajectory_steps(Path(episode.jsonl_path)))
        scanned_steps += len(records)
        for record in records:
            before, after = record.get("state_hash_before", ""), record.get("state_hash_after", "")
            # Dead-end: the action did not change state.
            if before and after and before == after:
                dead_end_steps += 1
            action = record.get("action", {})
            endpoint = action.get("endpoint", action.get("type", ""))
            if endpoint:
                action_counts[endpoint] += 1
        later_rewards = [r.get("reward", 0) for r in records[1:]]
        if later_rewards and all(r == 0.0 for r in later_rewards):
            zero_streak += 1
        if _drifted(records):
            drift += 1

    return {
        "total_episodes": total,
        "avg_reward": sum(rewards) / total,
        "avg_final_obj_score": sum(e.final_objective_score for e in episodes) / total,
        "avg_steps": avg_steps,
        "pass_rate": sum(1 for r in rewards if r > 0.5) / total,
        "termination_reasons": dict(term_reasons),
        "dead_end_rate": dead_end_steps / max(scanned_steps, 1),
        "reward_zero_streak_episodes": zero_streak,
        "state_drift_episodes": drift,
        "top_actions": sorted(action_counts.items(), key=lambda x: -x[1])[:10],
    }


_ROUNDING = {"avg_reward": 4, "avg_final_obj_score": 4, "avg_steps": 2, "pass_rate": 4, "dead_end_rate": 4}


def _rounded(stats: dict) -> dict:
    """Stats as reported. Issue rules read the unrounded values."""
    return {name: round(value, _ROUNDING[name]) if name in _ROUNDING else value for name, value in stats.items()}


def _drifted(records: list[dict]) -> bool:
    """State changed between steps without an action: step N starts where N-1 did not end."""
    for previous, current in zip(records, records[1:]):
        after, before = previous.get("state_hash_after", ""), current.get("state_hash_before", "")
        if after and before and after != before:
            return True
    return False


def _flag_container_issues(diagnosis: Diagnosis, stats: dict) -> None:
    total = stats["total_episodes"]
    avg_reward = stats["avg_reward"]
    if avg_reward < 0.1:
        diagnosis.flag(
            f"Extremely low avg reward ({avg_reward:.3f}) — objective scorer may be miscalibrated or environment is unsolvable",
            "Check /forge/state output: does it expose the fields the objective needs to evaluate progress?",
        )
    if stats["dead_end_rate"] > 0.3:
        diagnosis.flag(
            f"High dead-end rate ({stats['dead_end_rate']:.0%} of steps leave state unchanged) — actions not affecting state",
            "Inspect which endpoints have unchanged state hashes and verify the app's state bridge includes their effects in /forge/state",
        )
    diverged = stats["termination_reasons"].get("diverged", 0)
    if diverged / total > 0.5:
        diagnosis.flag(
            f"{diverged}/{total} episodes terminated due to 'diverged' — scorer drops to 0.0 prematurely",
            "The objective scorer gives 0.0 after some actions even though state changed — the objective may be too vague or the state representation too large for the LLM scorer to parse correctly",
        )
    zero_streak = stats["reward_zero_streak_episodes"]
    if zero_streak / max(total, 1) > 0.5:
        diagnosis.flag(
            f"{zero_streak}/{total} episodes have reward=0 for all steps after step 0 — reward signal is nearly zero-shot",
            "Only the first action ever gets non-zero reward, suggesting the scorer immediately plateaus at 0.0 after the first state change — the state bridge may be returning a stale snapshot",
        )
    drift = stats["state_drift_episodes"]
    if drift > 0:
        diagnosis.flag(
            f"State drift in {drift}/{total} episodes — state changes between steps without an action (non-determinism)",
            "Remove timestamps, UUIDs, or auto-incrementing IDs from /forge/state output; these make the reward signal noisy across seeds",
        )
    if stats["avg_steps"] < 5:
        diagnosis.flag(
            f"Episodes terminate very early (avg {stats['avg_steps']:.1f} steps) — environment may be too hard or dead-end detection too aggressive",
            "Consider raising dead_end_patience or max_steps for this environment type",
        )


def render(diagnosis: Diagnosis) -> None:
    typer.echo(f"\nDiagnosis: {diagnosis.env_name}\n{'─' * 50}")
    stats = diagnosis.container
    if stats:
        total = stats["total_episodes"]
        typer.echo(f"\nContainer episodes ({total})")
        typer.echo(f"  Pass rate:       {stats['pass_rate']:.0%}")
        typer.echo(f"  Avg reward:      {stats['avg_reward']:+.3f}")
        typer.echo(f"  Avg obj score:   {stats['avg_final_obj_score']:.3f}")
        typer.echo(f"  Avg steps:       {stats['avg_steps']:.1f}")
        typer.echo(f"  Dead-end rate:   {stats['dead_end_rate']:.0%}  (steps where action had no effect)")
        typer.echo(f"  Zero-streak eps: {stats['reward_zero_streak_episodes']}/{total}")
        typer.echo(f"  State drift eps: {stats['state_drift_episodes']}/{total}")
        typer.echo("\n  Termination reasons:")
        for reason, count in sorted(stats["termination_reasons"].items(), key=lambda x: -x[1]):
            typer.echo(f"    {count:3d}×  {reason}")
        if stats["top_actions"]:
            typer.echo("\n  Most-called endpoints:")
            for endpoint, count in stats["top_actions"][:5]:
                typer.echo(f"    {count:3d}×  {endpoint}")

    if diagnosis.gym:
        typer.echo(f"\nGym episodes ({diagnosis.gym['total_episodes']})")
        typer.echo(f"  Pass rate:  {diagnosis.gym['pass_rate']:.0%}")
        typer.echo(f"  Avg reward: {diagnosis.gym['avg_reward']:+.3f}")

    if diagnosis.issues:
        typer.echo(f"\nIssues found ({len(diagnosis.issues)})")
        for i, issue in enumerate(diagnosis.issues, 1):
            typer.echo(f"  {i}. {issue}")
        typer.echo("\nRecommendations")
        for i, rec in enumerate(diagnosis.recommendations, 1):
            typer.echo(f"  {i}. {rec}")
    else:
        typer.echo("\n✓ No quality issues detected")
    typer.echo("")
