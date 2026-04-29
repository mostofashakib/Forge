from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class RewardConfig:
    base_success: float = 1.0
    step_penalty: float = 0.01
    policy_violation_penalty: float = 1.0
    max_reward: float = 1.0
    min_reward: float = -1.0
    semantic_weight: float = 0.0
    invalid_action_penalty: float = 0.5


@dataclass
class ObservationConfig:
    mode: str = "full"
    actor_role: str = "agent"
    visible_entities: list[str] = field(default_factory=list)
    hidden_entities: list[str] = field(default_factory=list)


@dataclass
class EnvConfig:
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)


def load_config(pkg_dir: Path) -> EnvConfig:
    config_path = pkg_dir / "custom" / "config.yaml"
    if not config_path.exists():
        return EnvConfig()
    raw = yaml.safe_load(config_path.read_text()) or {}
    reward_raw = raw.get("reward", {})
    obs_raw = raw.get("observation", {})
    return EnvConfig(
        reward=RewardConfig(
            base_success=float(reward_raw.get("base_success", 1.0)),
            step_penalty=float(reward_raw.get("step_penalty", 0.01)),
            policy_violation_penalty=float(reward_raw.get("policy_violation_penalty", 1.0)),
            max_reward=float(reward_raw.get("max_reward", 1.0)),
            min_reward=float(reward_raw.get("min_reward", -1.0)),
            semantic_weight=float(reward_raw.get("semantic_weight", 0.0)),
            invalid_action_penalty=float(reward_raw.get("invalid_action_penalty", 0.5)),
        ),
        observation=ObservationConfig(
            mode=obs_raw.get("mode", "full"),
            actor_role=obs_raw.get("actor_role", "agent"),
            visible_entities=list(obs_raw.get("visible_entities", [])),
            hidden_entities=list(obs_raw.get("hidden_entities", [])),
        ),
    )
