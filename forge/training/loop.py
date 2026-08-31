"""End-to-end policy iteration over Forge environments."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from forge.training.checkpoint import PolicyCheckpoint
from forge.training.trainer import PolicyTrainer, TrainingConfig, TrainingResult


class ExperienceCollector(Protocol):
    """Collect graded policy experience and export trainer-ready files."""

    def collect(self, agent: object, output_dir: Path) -> Path:
        """Return a directory containing Forge's standard training exports."""


class PolicyLoader(Protocol):
    def __call__(self, checkpoint_dir: Path, environment: object | None) -> object: ...


@dataclass
class PolicyIterationConfig:
    training: TrainingConfig
    iterations: int = 1
    collect_after_update: bool = True

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")


@dataclass
class PolicyIterationResult:
    agent: object
    training_runs: list[TrainingResult] = field(default_factory=list)
    collection_dirs: list[Path] = field(default_factory=list)
    final_collection_dir: Path | None = None


class PolicyIterationLoop:
    """Collect → train → reload, then collect with the updated policy.

    The collector owns environment-specific rollout and export mechanics. This
    orchestrator owns only stage ordering and the policy feedback edge, making
    it usable by in-process, container, browser, and CLI collectors alike.
    """

    def __init__(
        self,
        collector: ExperienceCollector,
        trainer: PolicyTrainer | None = None,
        loader: PolicyLoader | None = None,
    ) -> None:
        self._collector = collector
        self._trainer = trainer or PolicyTrainer()
        self._loader = loader or _load_policy

    def run(
        self,
        initial_agent: object,
        config: PolicyIterationConfig,
        *,
        environment: object | None = None,
    ) -> PolicyIterationResult:
        root = Path(config.training.output_dir)
        current_agent = initial_agent
        current_base_model = config.training.base_model
        result = PolicyIterationResult(agent=current_agent)

        for index in range(config.iterations):
            iteration_dir = root / f"iteration-{index + 1:03d}"
            data_dir = Path(
                self._collector.collect(current_agent, iteration_dir / "experience")
            )
            result.collection_dirs.append(data_dir)
            training_result = self._trainer.train(replace(
                config.training,
                data_dir=data_dir,
                base_model=current_base_model,
                output_dir=iteration_dir / "checkpoint",
            ))
            result.training_runs.append(training_result)

            checkpoint_dir = Path(training_result.checkpoint_path)
            current_agent = self._loader(checkpoint_dir, environment)
            current_base_model = PolicyCheckpoint.load(checkpoint_dir).model_path

        result.agent = current_agent
        if config.collect_after_update:
            result.final_collection_dir = Path(
                self._collector.collect(current_agent, root / "updated-policy-experience")
            )
        return result


def _load_policy(checkpoint_dir: Path, environment: object | None) -> object:
    from forge.runtime.policy_loader import load_policy_agent

    return load_policy_agent(checkpoint_dir, environment=environment)
