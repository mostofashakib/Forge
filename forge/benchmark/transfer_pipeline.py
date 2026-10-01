"""Transfer benchmark: fine-tune on Forge's training envs, measure on held-out envs.

The experiment YAML named by ``eval_suite`` fixes the split. The base model is
trained only on demonstrations from ``train_envs``, then the checkpoint runs on
the disjoint ``heldout_envs`` once per seed. pass@k is estimated per task from
every decided sample across seeds, so transfer is measured on environments the
policy never saw during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from forge.benchmark.metrics import compute_pass_at_k
from forge.experiments import ExperimentConfig
from forge.training.trainer import TrainingConfig, TrainingObjective, TrainingResult

# Largest k reported. Every pass@k is estimated over the same tasks, which
# need at least this many decided samples each.
_MAX_K = 3


@dataclass
class TransferConfig:
    data_dir: Path
    # None trains the experiment's own base model.
    base_model: str | None
    output_dir: Path
    eval_suite: str = "experiments/internal_heldout.yaml"
    max_train_steps: int = 1000
    seeds: int = 3
    run_id: str = "transfer"
    objective: TrainingObjective = TrainingObjective.SFT


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


class Trainer(Protocol):
    def train(self, config: TrainingConfig) -> TrainingResult: ...


class SuiteEvaluator(Protocol):
    """Evaluates a checkpoint on an experiment's held-out split for one seed."""

    def __call__(self, model_path: str, suite: str, *, seed: int, runs_dir: Path, run_id: str) -> dict: ...


class TransferPipeline:
    """Train once, evaluate per seed, and estimate pass@k from the pooled samples."""

    def __init__(self, *, trainer: Trainer, evaluate: SuiteEvaluator) -> None:
        self._trainer = trainer
        self._evaluate = evaluate

    def __call__(self, config: TransferConfig) -> TransferResult:
        experiment = ExperimentConfig.load(config.eval_suite)
        seeds = _evaluation_seeds(config, experiment)
        run_dir = Path(config.output_dir) / config.run_id
        training = self._trainer.train(TrainingConfig(
            data_dir=Path(config.data_dir),
            base_model=experiment.base_model,
            output_dir=run_dir / "checkpoint",
            objective=config.objective,
            max_steps=config.max_train_steps,
            train_envs=experiment.train_envs,
            experiment_config=experiment.model_dump(mode="json"),
            seed=seeds[0],
            run_id=config.run_id,
        ))
        counts: dict[str, dict[str, int]] = {}
        for seed in seeds:
            evaluation = self._evaluate(
                training.checkpoint_path, config.eval_suite,
                seed=seed, runs_dir=run_dir / "runs", run_id=f"{config.run_id}-seed-{seed}",
            )
            _pool(counts, evaluation["task_pass_counts"])
        return _transfer_result(training.checkpoint_path, config.eval_suite, counts)


def run_transfer_pipeline(config: TransferConfig) -> TransferResult:
    """Production transfer: GPU training backends and container episodes."""
    from forge.benchmark._eval import evaluate_on_suite
    from forge.training.trainer import PolicyTrainer

    return TransferPipeline(trainer=PolicyTrainer(), evaluate=evaluate_on_suite)(config)


def _evaluation_seeds(config: TransferConfig, experiment: ExperimentConfig) -> list[int]:
    """The seeds to evaluate, after checking the request against the experiment."""
    if config.base_model is not None and config.base_model != experiment.base_model:
        raise ValueError(
            f"base model {config.base_model!r} does not match the experiment base model "
            f"{experiment.base_model!r} in {config.eval_suite}"
        )
    if config.seeds < 1:
        raise ValueError(f"transfer needs at least one seed, got {config.seeds}")
    if config.seeds > len(experiment.seeds):
        raise ValueError(
            f"requested {config.seeds} seeds but {config.eval_suite} declares "
            f"{len(experiment.seeds)}: {experiment.seeds}"
        )
    samples = config.seeds * experiment.determinism_repeats
    if samples < _MAX_K:
        raise ValueError(
            f"pass@{_MAX_K} needs at least {_MAX_K} samples per task, but {config.seeds} "
            f"seed(s) x {experiment.determinism_repeats} repeats gives {samples}; raise seeds"
        )
    return experiment.seeds[: config.seeds]


def _pool(totals: dict[str, dict[str, int]], counts: dict[str, dict[str, int]]) -> None:
    for task, task_counts in counts.items():
        pooled = totals.setdefault(task, {"decided": 0, "passed": 0})
        pooled["decided"] += task_counts["decided"]
        pooled["passed"] += task_counts["passed"]


def _transfer_result(model_path: str, suite: str, counts: dict[str, dict[str, int]]) -> TransferResult:
    decided = sum(task["decided"] for task in counts.values())
    if decided == 0:
        raise ValueError("the held-out split produced no decided episodes")
    estimable = [task for task in counts.values() if task["decided"] >= _MAX_K]
    if not estimable:
        raise ValueError(f"no held-out task has the {_MAX_K} decided samples pass@{_MAX_K} needs")
    return TransferResult(
        model_path=model_path,
        eval_suite=suite,
        task_completion_rate=sum(task["passed"] for task in counts.values()) / decided,
        success_at_1=_mean_pass_at_k(estimable, 1),
        success_at_3=_mean_pass_at_k(estimable, 3),
        num_eval_tasks=len(counts),
    )


def _mean_pass_at_k(tasks: list[dict[str, int]], k: int) -> float:
    return sum(compute_pass_at_k(task["decided"], task["passed"], k) for task in tasks) / len(tasks)
