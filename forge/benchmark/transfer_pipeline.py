"""Deferred external transfer-benchmark integration.

Forge's first evaluation path is the internal held-out protocol in ``_eval``.
External benchmark harnesses belong here later, without weakening that split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TransferConfig:
    data_dir: Path
    base_model: str
    output_dir: Path
    eval_suite: str = "external-deferred"
    max_train_steps: int = 1000


@dataclass
class TransferResult:
    model_path: str
    eval_suite: str
    task_completion_rate: float
    success_at_1: float
    success_at_3: float
    num_eval_tasks: int


def run_transfer_pipeline(config: TransferConfig) -> TransferResult:
    """Keep external suites out of the internal generalization metric for now."""
    raise NotImplementedError(
        "external transfer evaluation is intentionally deferred; use "
        "'forge train --experiment ...' followed by "
        "'forge benchmark eval --experiment ...' for internal held-out evaluation"
    )
