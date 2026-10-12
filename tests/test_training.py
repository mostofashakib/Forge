"""Policy training consumes real demonstrations and reports recorded results."""
import json

import pytest

from forge.training import AutotuneTrainer
from forge.contracts.checkpoint import PolicyCheckpoint
from forge.contracts.training import NoTrainingSignalError, TrainingConfig


class RecordingBackend:
    def __init__(self):
        self.examples = []

    def train(self, *, base_model, examples, output_dir, max_steps):
        self.examples = examples
        return output_dir / "trained_model"


def training_config(tmp_path, rows, **overrides):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sft_pairs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    return TrainingConfig(
        data_dir=data_dir, output_dir=tmp_path / "out", base_model="base",
        **overrides,
    )


def test_training_uses_real_examples_and_measured_checkpoint_metadata(tmp_path):
    config = training_config(tmp_path, [
        {"prompt": "Task: finish\nEnvironment: alpha", "completion": "submit()", "total_reward": 0.4},
        {"messages": [{"role": "user", "content": "Task: save\nEnvironment: alpha"},
                      {"role": "assistant", "content": "save()"}], "total_reward": 0.8},
    ], training_mode="offline", train_envs=["alpha"])
    backend = RecordingBackend()

    result = AutotuneTrainer(backend).train(config)

    assert [(example.prompt, example.completion) for example in backend.examples] == [
        ("Task: finish\nEnvironment: alpha", "submit()"),
        ("Task: save\nEnvironment: alpha", "save()"),
    ]
    checkpoint = PolicyCheckpoint.load(config.output_dir)
    assert result.num_examples == checkpoint.num_examples == 2
    assert result.mean_reward == checkpoint.mean_reward == pytest.approx(0.6)
    assert checkpoint.model_path == str(tmp_path / "out" / "trained_model")
    assert checkpoint.training_mode == "offline"


@pytest.mark.parametrize("rows", [
    [],
    [{"prompt": "Task: finish\nEnvironment: alpha_extra", "completion": "wrong()"}],
    [{"prompt": "Task: alpha\nEnvironment: heldout", "completion": "leak()"}],
    [{"messages": []}],
])
def test_missing_matching_demonstrations_never_create_synthetic_examples(tmp_path, rows):
    config = training_config(tmp_path, rows, train_envs=["alpha"])
    with pytest.raises(NoTrainingSignalError):
        AutotuneTrainer(RecordingBackend()).train(config)
    assert not (tmp_path / "out" / "policy_checkpoint.json").exists()


def test_training_without_backend_does_not_report_a_successful_checkpoint(tmp_path):
    config = training_config(tmp_path, [{"prompt": "task", "completion": "submit()"}])
    with pytest.raises(RuntimeError, match="backend"):
        AutotuneTrainer().train(config)
    assert not (tmp_path / "out").exists()


def test_corrupt_training_data_fails_with_file_and_line_context(tmp_path):
    config = training_config(tmp_path, [{"prompt": "task", "completion": "submit()"}])
    with (tmp_path / "data" / "sft_pairs.jsonl").open("a") as handle:
        handle.write("\n{broken\n")
    with pytest.raises(ValueError, match=r"sft_pairs.jsonl.*2"):
        AutotuneTrainer(RecordingBackend()).train(config)
    assert not (tmp_path / "out").exists()


def test_training_does_not_treat_unrelated_files_as_training_signal(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset_manifest.json").write_text('{"files": []}')
    config = TrainingConfig(data_dir=data_dir, output_dir=tmp_path / "out", base_model="base")
    with pytest.raises(NoTrainingSignalError):
        AutotuneTrainer(RecordingBackend()).train(config)


@pytest.mark.parametrize("objective", ["grpo", "dpo", "ppo"])
def test_sft_backend_rejects_other_objectives_without_writing_a_checkpoint(tmp_path, objective):
    config = training_config(tmp_path, [{"prompt": "task", "completion": "submit()"}], objective=objective)
    backend = RecordingBackend()
    with pytest.raises(NotImplementedError, match=objective):
        AutotuneTrainer(backend).train(config)
    assert backend.examples == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("model_path", [None, "", {}, 42])
def test_invalid_backend_results_never_create_a_checkpoint(tmp_path, model_path):
    config = training_config(tmp_path, [{"prompt": "task", "completion": "submit()"}])

    class InvalidBackend:
        def train(self, **kwargs):
            return model_path

    with pytest.raises(ValueError, match="model path"):
        AutotuneTrainer(InvalidBackend()).train(config)
    assert not (tmp_path / "out" / "policy_checkpoint.json").exists()


def test_backend_errors_propagate_without_writing_a_checkpoint(tmp_path):
    config = training_config(tmp_path, [{"prompt": "task", "completion": "submit()"}])

    class FailingBackend:
        def train(self, **kwargs):
            raise RuntimeError("training ran out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        AutotuneTrainer(FailingBackend()).train(config)
    assert not (tmp_path / "out" / "policy_checkpoint.json").exists()
