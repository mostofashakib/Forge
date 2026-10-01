"""Transfer: fine-tune on the training envs, then measure on disjoint held-out envs."""
from __future__ import annotations

import json
from functools import partial

import pytest

from forge.benchmark._eval import EpisodeOutcome, evaluate_on_suite
from forge.benchmark.task_suite import Task
from forge.benchmark.transfer_pipeline import TransferConfig, TransferPipeline
from forge.training.trainer import NoTrainingSignalError, PolicyTrainer


class _RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def train(self, base_model, examples, output_dir, max_steps):
        self.calls.append({"base_model": base_model, "examples": examples, "max_steps": max_steps})
        return str(output_dir / "forge_policy")


class _Provider:
    def tasks_for(self, domain: str, depth: int) -> list[Task]:
        return [Task(
            name=f"{domain}_task", domain=domain, objective="finish",
            success_fn=lambda _state: False, difficulty=1,
        )]


class _Runner:
    """Plays back a fixed outcome list per held-out env and records each call."""

    def __init__(self, outcomes: dict[str, list[EpisodeOutcome]]) -> None:
        self._outcomes = {env: iter(items) for env, items in outcomes.items()}
        self.calls: list[tuple[str, int]] = []

    def __call__(self, task, seed, path):
        self.calls.append((task.domain, seed))
        return next(self._outcomes[task.domain])


def _experiment(tmp_path, *, seeds="[7, 8]", base_model="base", extra=""):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "train_envs: [train_a]\nheldout_envs: [held_a, held_b]\n"
        f"reward_preset: full_layered_partial\nbase_model: {base_model}\n"
        f"seeds: {seeds}\n{extra}",
        encoding="utf-8",
    )
    return path


def _training_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"prompt": "Environment: train_a\nfile the report", "completion": "submit()"},
        {"prompt": "Environment: held_a\nleaked demo", "completion": "cheat()"},
    ]
    (data_dir / "sft_pairs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return data_dir


def _config(tmp_path, experiment, **overrides):
    values = {
        "data_dir": tmp_path / "data",
        "base_model": "base",
        "output_dir": tmp_path / "out",
        "eval_suite": str(experiment),
        "max_train_steps": 40,
        "seeds": 2,
        "run_id": "transfer_1",
    }
    values.update(overrides)
    return TransferConfig(**values)


def _pipeline(backend, runner):
    return TransferPipeline(
        trainer=PolicyTrainer(backend=backend),
        evaluate=partial(evaluate_on_suite, task_provider=_Provider(), episode_runner=runner),
    )


def _passed(flag: bool) -> EpisodeOutcome:
    return EpisodeOutcome(passed=flag, reward=1.0 if flag else 0.0)


def test_transfer_trains_on_train_envs_and_scores_heldout_tasks_with_pass_at_k(tmp_path):
    experiment = _experiment(tmp_path)
    _training_data(tmp_path)
    backend = _RecordingBackend()
    # Two seeds times two determinism repeats: four samples per task.
    runner = _Runner({
        "held_a": [_passed(True), _passed(False), _passed(True), _passed(False)],
        "held_b": [_passed(False)] * 4,
    })

    result = _pipeline(backend, runner)(_config(tmp_path, experiment))

    # Only the training env's demonstration reaches the update.
    assert [call["max_steps"] for call in backend.calls] == [40]
    assert [example.prompt for example in backend.calls[0]["examples"]] == [
        "Environment: train_a\nfile the report"
    ]
    assert {seed for _, seed in runner.calls} == {7, 8}
    assert {env for env, _ in runner.calls} == {"held_a", "held_b"}
    assert result.model_path == str(tmp_path / "out" / "transfer_1" / "checkpoint")
    assert result.eval_suite == str(experiment)
    assert result.num_eval_tasks == 2
    assert result.task_completion_rate == 0.25
    # held_a: c=2 of n=4 gives pass@1 0.5 and pass@3 1.0. held_b never passes.
    assert result.success_at_1 == 0.25
    assert result.success_at_3 == 0.5


def test_each_seed_writes_its_own_result_record(tmp_path):
    experiment = _experiment(tmp_path)
    _training_data(tmp_path)
    runner = _Runner({"held_a": [_passed(True)] * 4, "held_b": [_passed(True)] * 4})

    _pipeline(_RecordingBackend(), runner)(_config(tmp_path, experiment))

    runs = tmp_path / "out" / "transfer_1" / "runs"
    seeds = [json.loads((runs / f"transfer_1-seed-{seed}" / "result.json").read_text())["seed"] for seed in (7, 8)]
    assert seeds == [7, 8]


def test_abstentions_do_not_count_as_failed_samples(tmp_path):
    experiment = _experiment(tmp_path, seeds="[7, 8]", extra="max_abstention_rate: 0.5\n")
    _training_data(tmp_path)
    undecided = EpisodeOutcome(passed=False, reward=0.0, indeterminate=True)
    runner = _Runner({
        "held_a": [_passed(True), _passed(True), _passed(True), undecided],
        "held_b": [_passed(True)] * 4,
    })

    result = _pipeline(_RecordingBackend(), runner)(_config(tmp_path, experiment))

    assert result.task_completion_rate == 1.0
    assert result.success_at_1 == 1.0


def test_tasks_with_fewer_than_three_decided_samples_are_left_out_of_pass_at_k(tmp_path):
    experiment = _experiment(tmp_path, seeds="[7, 8]", extra="max_abstention_rate: 0.5\n")
    _training_data(tmp_path)
    undecided = EpisodeOutcome(passed=False, reward=0.0, indeterminate=True)
    # held_a has two decided samples, too few to estimate pass@3 honestly.
    runner = _Runner({
        "held_a": [_passed(True), undecided, _passed(True), undecided],
        "held_b": [_passed(False)] * 4,
    })

    result = _pipeline(_RecordingBackend(), runner)(_config(tmp_path, experiment))

    assert result.success_at_1 == 0.0
    assert result.success_at_3 == 0.0
    assert result.task_completion_rate == pytest.approx(2 / 6)


def test_an_unset_base_model_trains_the_experiments_base_model(tmp_path):
    experiment = _experiment(tmp_path, base_model="org/declared")
    _training_data(tmp_path)
    backend = _RecordingBackend()
    runner = _Runner({"held_a": [_passed(True)] * 4, "held_b": [_passed(True)] * 4})

    _pipeline(backend, runner)(_config(tmp_path, experiment, base_model=None))

    assert backend.calls[0]["base_model"] == "org/declared"


def test_a_base_model_other_than_the_experiments_fails_before_training(tmp_path):
    experiment = _experiment(tmp_path, base_model="base")
    _training_data(tmp_path)
    backend = _RecordingBackend()

    with pytest.raises(ValueError, match="does not match the experiment base model"):
        _pipeline(backend, _Runner({}))(_config(tmp_path, experiment, base_model="other"))
    assert backend.calls == []


def test_more_seeds_than_the_experiment_declares_fails_before_training(tmp_path):
    experiment = _experiment(tmp_path, seeds="[7, 8]")
    _training_data(tmp_path)
    backend = _RecordingBackend()

    with pytest.raises(ValueError, match="declares 2"):
        _pipeline(backend, _Runner({}))(_config(tmp_path, experiment, seeds=3))
    assert backend.calls == []


def test_too_few_samples_for_pass_at_3_fails_before_training(tmp_path):
    experiment = _experiment(tmp_path, seeds="[7]")
    _training_data(tmp_path)
    backend = _RecordingBackend()

    with pytest.raises(ValueError, match="pass@3"):
        _pipeline(backend, _Runner({}))(_config(tmp_path, experiment, seeds=1))
    assert backend.calls == []


def test_missing_experiment_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="experiment config not found"):
        _pipeline(_RecordingBackend(), _Runner({}))(
            _config(tmp_path, tmp_path / "missing.yaml")
        )


def test_no_training_data_fails_without_evaluating(tmp_path):
    experiment = _experiment(tmp_path)
    (tmp_path / "data").mkdir()
    runner = _Runner({})

    with pytest.raises(NoTrainingSignalError):
        _pipeline(_RecordingBackend(), runner)(_config(tmp_path, experiment))
    assert runner.calls == []
