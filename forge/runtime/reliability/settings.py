"""Retry and snapshot settings, read from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_RETRY_CAP = 3
DEFAULT_SHORT_EPISODE_THRESHOLD = 5
DEFAULT_SNAPSHOT_INTERVAL = 5


@dataclass(frozen=True)
class ReliabilitySettings:
    retry_cap: int = DEFAULT_RETRY_CAP
    short_episode_threshold: int = DEFAULT_SHORT_EPISODE_THRESHOLD
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL


def get_reliability_settings() -> ReliabilitySettings:
    """Read reliability settings from environment / saved config."""
    try:
        retry_cap = int(os.environ.get("FORGE_RETRY_CAP", DEFAULT_RETRY_CAP))
    except ValueError:
        retry_cap = DEFAULT_RETRY_CAP

    try:
        short_threshold = int(
            os.environ.get("FORGE_SHORT_EPISODE_THRESHOLD", DEFAULT_SHORT_EPISODE_THRESHOLD)
        )
    except ValueError:
        short_threshold = DEFAULT_SHORT_EPISODE_THRESHOLD

    try:
        snapshot_interval = int(
            os.environ.get("FORGE_SNAPSHOT_INTERVAL", DEFAULT_SNAPSHOT_INTERVAL)
        )
    except ValueError:
        snapshot_interval = DEFAULT_SNAPSHOT_INTERVAL

    return ReliabilitySettings(
        retry_cap=max(0, retry_cap),
        short_episode_threshold=max(1, short_threshold),
        snapshot_interval=max(1, snapshot_interval),
    )
