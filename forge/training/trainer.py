"""Train Forge's own policy from its own graded rollouts.

Closes the RL loop: the runtime collects and grades episodes
(`ForgeEnv` / `LayeredVerifier` / `RewardBreakdown`), the export writers emit
`grpo_rollouts.parquet` and `preference_pairs.jsonl`, and this trainer turns
those grades into a policy update — either group-relative-advantage **GRPO** over
rollouts or preference-optimization **DPO** over chosen/rejected pairs.

This is distinct from task #1 (`forge/benchmark/`): that is zero-shot transfer
*evaluation* of a base model on external suites; this is *training* Forge's own
policy from its own scored experience, and produces a checkpoint the runtime
agents load via `forge.training.checkpoint.load_policy_agent`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from forge.training._backends import DPOBackend, GRPOBackend, TrainingBackend
from forge.training.checkpoint import PolicyCheckpoint
from forge.training.dataset import load_preferences, load_rollouts
from forge.training.reward_mapping import dpo_examples, grpo_advantages


class TrainingObjective(str, Enum):
    GRPO = "grpo"
    DPO = "dpo"


class NoTrainingSignalError(RuntimeError):
    """The graded data carries no learnable signal, so no update is produced.

    Raised for an empty/absent export, an all-failing or all-equal rollout set
    (no group-relative advantage), or preference pairs that are all ties.
    """


@dataclass
class TrainingConfig:
    data_dir: Path
    base_model: str
    output_dir: Path
    objective: TrainingObjective = TrainingObjective.GRPO
    max_steps: int = 500


@dataclass
class TrainingResult:
    checkpoint_path: str
    objective: str
    num_examples: int
    mean_reward: float


_ROLLOUTS_FILE = "grpo_rollouts.parquet"
_PREFERENCES_FILE = "preference_pairs.jsonl"


class PolicyTrainer:
    """Loads graded rollouts, maps rewards to a signal, and trains a checkpoint."""

    def __init__(self, backend: TrainingBackend | None = None) -> None:
        # A backend may be injected (e.g. for tests); otherwise the objective's
        # default GPU-gated backend is used.
        self._backend = backend

    def train(self, config: TrainingConfig) -> TrainingResult:
        objective = TrainingObjective(config.objective)
        examples, mean_reward = self._prepare(objective, config.data_dir)
        if not examples:
            raise NoTrainingSignalError(
                f"no {objective.value} training signal in {config.data_dir}: "
                "the graded rollouts are empty, all-failing, or carry no relative signal"
            )

        backend = self._backend or self._default_backend(objective)
        model_path = backend.train(
            base_model=config.base_model,
            examples=examples,
            output_dir=Path(config.output_dir),
            max_steps=config.max_steps,
        )

        checkpoint = PolicyCheckpoint(
            objective=objective.value,
            base_model=config.base_model,
            model_path=model_path,
            num_examples=len(examples),
            mean_reward=mean_reward,
        )
        checkpoint.save(Path(config.output_dir))
        return TrainingResult(
            checkpoint_path=str(config.output_dir),
            objective=objective.value,
            num_examples=len(examples),
            mean_reward=mean_reward,
        )

    # ------------------------------------------------------------------

    def _prepare(self, objective: TrainingObjective, data_dir: Path) -> tuple[list, float]:
        data_dir = Path(data_dir)
        if objective is TrainingObjective.GRPO:
            path = data_dir / _ROLLOUTS_FILE
            if not path.exists():
                return [], 0.0
            rollouts = load_rollouts(path)
            examples = grpo_advantages(rollouts)
            mean = sum(r.total_reward for r in rollouts) / len(rollouts) if rollouts else 0.0
            return examples, mean

        path = data_dir / _PREFERENCES_FILE
        if not path.exists():
            return [], 0.0
        preferences = load_preferences(path)
        examples = dpo_examples(preferences)
        rewards = [p.chosen_reward for p in preferences] + [p.rejected_reward for p in preferences]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        return examples, mean

    def _default_backend(self, objective: TrainingObjective) -> TrainingBackend:
        return GRPOBackend() if objective is TrainingObjective.GRPO else DPOBackend()
