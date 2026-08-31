"""Backward-compatibility shim for the pre-contracts episode module.

The classes that used to live here now live in `forge.contracts.episode` and
`forge.contracts.termination`. This module just re-exports those names so
existing imports keep working; new code should import from `forge.contracts`
directly.
"""
from __future__ import annotations

# Moved to forge/contracts/episode.py and forge/contracts/termination.py.
# Re-exported here so existing imports keep working; prefer importing from
# forge.contracts.
from forge.contracts.episode import (  # noqa: F401
    BaseEpisodeConfig,
    BaseEpisodeResult,
    TrajectoryWriter,
)
from forge.contracts.termination import (  # noqa: F401
    MaxStepsTerminationPolicy,
    TerminationMonitor,
    ThresholdTerminationPolicy,
)
