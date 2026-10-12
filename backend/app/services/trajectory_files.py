"""Read and stream recorded agent trajectories without buffering entire files."""
from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from forge.jsonl import iter_jsonl_objects

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrajectoryData:
    steps: list[dict]
    summary: dict | None


def read_trajectory(path: Path, *, max_steps: int | None = None) -> TrajectoryData:
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    steps: deque[dict] = deque(maxlen=max_steps)
    summary = None
    for record in iter_jsonl_objects(path):
        if record.get("type") == "episode_summary":
            summary = record
        else:
            steps.append(record)
    return TrajectoryData(steps=list(steps), summary=summary)


def iter_trajectory_lines(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield line.strip() + "\n"
        except FileNotFoundError:
            logger.warning("Trajectory file is missing during export: %s", path)
        except OSError:
            logger.exception("Could not export trajectory %s", path)
            raise
