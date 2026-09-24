"""Per-environment reward settings stored in <env>/reward_config.json."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from forge.reward_presets import RewardPreset
from forge.settings import generated_envs_root

logger = logging.getLogger(__name__)

_DEFAULT_PRESET = RewardPreset.FULL_LAYERED_PARTIAL.value


@dataclass(frozen=True)
class RewardConfig:
    scoring_methods: list[str] = field(default_factory=lambda: ["llm"])
    reward_preset: str = _DEFAULT_PRESET


def _config_path(env_name: str) -> Path:
    return generated_envs_root() / env_name / "reward_config.json"


def _read_raw(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[reward-config] ignoring unreadable %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("[reward-config] ignoring %s: expected an object", path)
        return {}
    return data


def load_reward_config(env_name: str) -> RewardConfig:
    data = _read_raw(_config_path(env_name))
    if "scoring_methods" in data:
        methods = data["scoring_methods"] or ["llm"]
    elif "scoring_method" in data:  # single-string format from older builds
        methods = [data["scoring_method"]]
    else:
        methods = ["llm"]
    try:
        preset = RewardPreset(data.get("reward_preset", _DEFAULT_PRESET)).value
    except ValueError:
        logger.warning(
            "[reward-config] unknown preset %r for %s; using %s",
            data.get("reward_preset"), env_name, _DEFAULT_PRESET,
        )
        preset = _DEFAULT_PRESET
    return RewardConfig(scoring_methods=list(methods), reward_preset=preset)


def save_reward_config(
    env_name: str,
    *,
    scoring_methods: list[str] | None = None,
    reward_preset: str | None = None,
) -> None:
    """Update the given fields and keep the rest."""
    current = load_reward_config(env_name)
    path = _config_path(env_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "reward_preset": RewardPreset(reward_preset or current.reward_preset).value,
        "scoring_methods": scoring_methods or current.scoring_methods,
    }), encoding="utf-8")
