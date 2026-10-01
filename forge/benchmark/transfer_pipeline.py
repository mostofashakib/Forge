"""Deferred external transfer-benchmark integration.

Forge's first evaluation path is the internal held-out protocol in ``_eval``.
External benchmark harnesses belong here later, without weakening that split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class TransferConfig:
    data_dir: Path
    base_model: str
    output_dir: Path
    eval_suite: str = "external-deferred"
    max_train_steps: int = 1000
    seeds: int = 3


@dataclass
class TransferResult:
    model_path: str
    eval_suite: str
    task_completion_rate: float
    success_at_1: float
    success_at_3: float
    num_eval_tasks: int

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "eval_suite": self.eval_suite,
            "task_completion_rate": self.task_completion_rate,
            "pass_at_1": self.success_at_1,
            "pass_at_3": self.success_at_3,
            "num_eval_tasks": self.num_eval_tasks,
        }


class TransferEvaluator(Protocol):
    """Runs a transfer evaluation and returns what it measured."""

    def __call__(self, config: TransferConfig) -> TransferResult: ...


def run_transfer_pipeline(config: TransferConfig) -> TransferResult:
    """Keep external suites out of the internal generalization metric for now."""
    raise NotImplementedError(
        "external transfer evaluation is intentionally deferred; use "
        "'forge train --experiment ...' followed by "
        "'forge benchmark eval --experiment ...' for internal held-out evaluation"
    )
