"""Train policies from recorded SFT demonstrations through an injected backend."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from forge.contracts.checkpoint import PolicyCheckpoint
from forge.contracts.training import NoTrainingSignalError, TrainingConfig, TrainingResult
from forge.jsonl import iter_jsonl_objects

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    completion: str
    reward: float | None = None


class TrainingBackend(Protocol):
    def train(
        self, *, base_model: str, examples: list[TrainingExample], output_dir: Path, max_steps: int,
    ) -> str | Path: ...


class AutotuneTrainer:
    def __init__(self, backend: TrainingBackend | None = None):
        self.backend = backend

    def train(self, config: TrainingConfig) -> TrainingResult:
        examples = _load_examples(Path(config.data_dir), set(config.train_envs or []))
        objective = getattr(config.objective, "value", str(config.objective))
        if not examples:
            raise NoTrainingSignalError(
                f"no {objective} training signal in {config.data_dir}: no matching SFT demonstrations"
            )
        if self.backend is None:
            raise RuntimeError("No training backend configured. Supply a TrainingBackend to AutotuneTrainer.")
        if objective != "sft":
            raise NotImplementedError(f"The SFT training backend does not support objective {objective!r}")

        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.backend.train(
            base_model=config.base_model, examples=examples,
            output_dir=out_dir, max_steps=config.max_steps,
        )
        if not isinstance(model_path, (str, Path)) or not str(model_path).strip():
            raise ValueError("Training backend returned no model path")
        rewards = [example.reward for example in examples if example.reward is not None]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        checkpoint = PolicyCheckpoint(
            objective=objective, training_mode=config.training_mode,
            base_model=config.base_model, model_path=str(model_path),
            num_examples=len(examples), mean_reward=mean_reward,
            experiment_config=config.experiment_config or {},
            train_envs=config.train_envs or [], seed=config.seed, run_id=config.run_id,
        )
        checkpoint.save(out_dir)
        return TrainingResult(
            checkpoint_path=str(out_dir), num_examples=len(examples),
            mean_reward=mean_reward, objective=objective,
        )


def _load_examples(data_dir: Path, train_envs: set[str]) -> list[TrainingExample]:
    path = data_dir / "sft_pairs.jsonl"
    if not path.exists():
        return []
    examples = []
    for record in iter_jsonl_objects(path):
        prompt, completion = record.get("prompt"), record.get("completion")
        if prompt is None and completion is None:
            messages = record.get("messages")
            if isinstance(messages, list) and len(messages) >= 2:
                prompt = messages[0].get("content") if isinstance(messages[0], dict) else None
                completion = messages[1].get("content") if isinstance(messages[1], dict) else None
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(completion, str) or not completion.strip():
            logger.warning("Skipping an SFT record without a prompt and completion in %s", path)
            continue
        environment = record.get("env_name")
        if environment is None:
            environment = next(
                (line.removeprefix("Environment:").strip() for line in prompt.splitlines()
                 if line.startswith("Environment:")), None,
            )
        if train_envs and environment not in train_envs:
            continue
        reward = record.get("total_reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            reward = None
        examples.append(TrainingExample(prompt=prompt, completion=completion, reward=reward))
    return examples
