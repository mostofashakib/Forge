from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

from forge.schema.state_schema import StateSchemaManifest
from forge.jsonl import iter_jsonl_objects

logger = logging.getLogger(__name__)


@dataclass
class EnvQualityMetrics:
    env_name: str
    state_coverage_score: float
    reward_density: float
    dead_end_rate: float
    action_diversity: float
    num_episodes: int
    num_steps: int


def compute_env_quality(
    episode_dir: Path,
    manifest: StateSchemaManifest,
) -> EnvQualityMetrics:
    """Compute quality metrics from JSONL episode files in episode_dir."""
    total_steps = 0
    steps_with_reward = 0
    coverage_sum = 0.0
    dead_end_episodes = 0
    endpoint_count = 0
    unique_endpoints: set[str] = set()
    num_episodes = 0

    for jsonl_path in episode_dir.rglob("*.jsonl"):
        episode_counted = False
        try:
            for rec in iter_jsonl_objects(jsonl_path, skip_invalid=True):
                if not episode_counted:
                    num_episodes += 1
                    episode_counted = True
                if rec.get("type") == "episode_summary":
                    if rec.get("termination_reason") == "dead_end":
                        dead_end_episodes += 1
                    continue

                total_steps += 1
                reward = rec.get("reward", 0.0)
                if reward > 0:
                    steps_with_reward += 1

                state_after = rec.get("state_after", {})
                coverage_sum += manifest.coverage_score(state_after)

                endpoint = rec.get("action", {}).get("endpoint", "")
                if endpoint:
                    endpoint_count += 1
                    unique_endpoints.add(endpoint)

            if not episode_counted and jsonl_path.stat().st_size:
                num_episodes += 1
        except (OSError, UnicodeError) as exc:
            logger.warning("Could not read %s: %s", jsonl_path, exc)

    if total_steps == 0:
        return EnvQualityMetrics(
            env_name=manifest.env_name,
            state_coverage_score=0.0,
            reward_density=0.0,
            dead_end_rate=0.0,
            action_diversity=0.0,
            num_episodes=num_episodes,
            num_steps=0,
        )

    return EnvQualityMetrics(
        env_name=manifest.env_name,
        state_coverage_score=coverage_sum / total_steps,
        reward_density=steps_with_reward / total_steps,
        dead_end_rate=dead_end_episodes / max(num_episodes, 1),
        action_diversity=len(unique_endpoints) / endpoint_count if endpoint_count else 0.0,
        num_episodes=num_episodes,
        num_steps=total_steps,
    )
