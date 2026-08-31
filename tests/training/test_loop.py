from pathlib import Path

import pytest

from forge.training.checkpoint import PolicyCheckpoint
from forge.training.loop import PolicyIterationConfig, PolicyIterationLoop
from forge.training.trainer import TrainingConfig, TrainingResult


class _Collector:
    def __init__(self) -> None:
        self.agents = []

    def collect(self, agent, output_dir: Path) -> Path:
        self.agents.append(agent)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


class _Trainer:
    def __init__(self) -> None:
        self.configs = []

    def train(self, config: TrainingConfig) -> TrainingResult:
        self.configs.append(config)
        model_path = config.output_dir / "model"
        model_path.mkdir(parents=True, exist_ok=True)
        PolicyCheckpoint(
            objective="grpo",
            base_model=config.base_model,
            model_path=str(model_path),
            num_examples=2,
            mean_reward=0.5,
        ).save(config.output_dir)
        return TrainingResult(
            checkpoint_path=str(config.output_dir),
            objective="grpo",
            num_examples=2,
            mean_reward=0.5,
        )


def test_policy_iteration_collects_trains_reloads_and_collects_again(tmp_path):
    collector = _Collector()
    trainer = _Trainer()
    loaded = []

    def loader(checkpoint_dir, environment):
        agent = f"updated:{checkpoint_dir.name}"
        loaded.append((checkpoint_dir, environment))
        return agent

    loop = PolicyIterationLoop(collector, trainer=trainer, loader=loader)
    environment = object()
    result = loop.run(
        "base-agent",
        PolicyIterationConfig(training=TrainingConfig(
            data_dir=tmp_path / "unused",
            base_model="base-model",
            output_dir=tmp_path / "run",
        )),
        environment=environment,
    )

    assert collector.agents == ["base-agent", "updated:checkpoint"]
    assert trainer.configs[0].data_dir.name == "experience"
    assert loaded[0][1] is environment
    assert result.agent == "updated:checkpoint"
    assert result.final_collection_dir.name == "updated-policy-experience"


def test_policy_iteration_rejects_zero_iterations(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        PolicyIterationConfig(
            training=TrainingConfig(
                data_dir=tmp_path,
                base_model="base",
                output_dir=tmp_path / "out",
            ),
            iterations=0,
        )


def test_policy_iteration_does_not_collect_updated_policy_when_disabled(tmp_path):
    collector = _Collector()
    loop = PolicyIterationLoop(
        collector,
        trainer=_Trainer(),
        loader=lambda _checkpoint, _environment: "updated",
    )
    result = loop.run(
        "base",
        PolicyIterationConfig(
            training=TrainingConfig(
                data_dir=tmp_path,
                base_model="base",
                output_dir=tmp_path / "out",
            ),
            collect_after_update=False,
        ),
    )

    assert collector.agents == ["base"]
    assert result.final_collection_dir is None
