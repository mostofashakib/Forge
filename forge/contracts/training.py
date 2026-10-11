"""Contracts for training jobs and signals."""
from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class TrainingObjective(str, Enum):
    GRPO = "grpo"
    DPO = "dpo"
    SFT = "sft"
    PPO = "ppo"


class NoTrainingSignalError(RuntimeError):
    """Raised when an exported dataset has no valid signal for policy improvement."""
    pass


from pathlib import Path


class TrainingConfig(BaseModel):
    """Specification of a training run."""
    model_config = {"arbitrary_types_allowed": True}

    data_dir: Path | str
    base_model: str
    output_dir: Path | str
    objective: TrainingObjective | str = "sft"
    training_mode: str = "online"
    max_steps: int = 500
    train_envs: list[str] | None = None
    experiment_config: dict | None = None
    seed: int | None = None
    run_id: str = ""


class TrainingResult(BaseModel):
    """Result of a policy training run."""
    checkpoint_path: str
    num_examples: int = 0
    mean_reward: float = 0.0
    objective: str = "sft"
