from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

Difficulty = int  # 1–5


@dataclass
class Task:
    """A single benchmark task, resolved from a generated environment's own
    compiled task templates (see :mod:`forge.benchmark.compiled_tasks`).

    ``success_fn`` is retained for the dataclass contract but is not called on
    the benchmark path — generated environments are graded inside the container
    episode runner by their own ``reward_fn``/verifier.
    """

    name: str
    domain: str
    objective: str
    success_fn: Callable[[dict], bool]
    difficulty: Difficulty
