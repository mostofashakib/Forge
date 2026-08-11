from __future__ import annotations

from forge.extraction.schemas import TaskTemplate
from dataclasses import dataclass
from typing import Callable

Difficulty = int  # 1–5


@dataclass
class Task:
    """A single benchmark task, resolved from a generated environment's own
    compiled task templates (see :mod:`forge.benchmark.compiled_tasks`).

    ``success_fn`` is retained for the dataclass contract but is not called on
    the benchmark path. Generated environments are graded structurally, against
    the ``template``'s compiled success and failure conditions.
    """

    name: str
    domain: str
    objective: str
    success_fn: Callable[[dict], bool]
    difficulty: Difficulty
    # The compiled task this was resolved from. Carries the success and
    # failure conditions the environment was built with, which are the
    # ground truth an episode is graded against.
    template: "TaskTemplate | None" = None
