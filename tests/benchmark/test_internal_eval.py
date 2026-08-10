from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from forge.benchmark._eval import (
    EpisodeOutcome,
    _PolicyContainerAdapter,
    _has_reward_hacking_pattern,
    evaluate_on_suite,
)
from forge.benchmark.task_suite import Task
from forge.training.checkpoint import PolicyCheckpoint


class _Provider:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def tasks_for(self, domain: str, depth: int) -> list[Task]:
        self.requested.append(domain)
        return [Task(
            name=f"{domain}_task", domain=domain, objective="finish",
            success_fn=lambda _state: False, difficulty=1,
        )]


def _experiment(path, *, train=None, heldout=None, seeds=None):
    path.write_text(
        "train_envs:\n" + "".join(f"  - {name}\n" for name in (train or ["train_a"]))
        + "heldout_envs:\n" + "".join(f"  - {name}\n" for name in (heldout or ["held_a", "held_b"]))
        + "reward_preset: balanced\nbase_model: base\n"
        + f"seeds: {seeds or [7]}\n",
        encoding="utf-8",
    )


def _config_dict(train=None, heldout=None, seeds=None):
    return {
        "train_envs": train or ["train_a"],
        "heldout_envs": heldout or ["held_a", "held_b"],
        "reward_preset": "balanced",
        "base_model": "base",
        "seeds": seeds or [7],
    }


def test_evaluate_runs_only_heldout_and_writes_result(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    _experiment(config_path)
    checkpoint_dir = tmp_path / "checkpoint"
    PolicyCheckpoint(
        objective="grpo", base_model="base", model_path="model", num_examples=2,
        mean_reward=0.5, train_envs=["train_a"], seed=7, run_id="run-7",
        experiment_config=_config_dict(),
    ).save(checkpoint_dir)
    provider = _Provider()
    outcomes = iter([
        EpisodeOutcome(passed=True, reward=1.0, reward_hacking=False),
        EpisodeOutcome(passed=False, reward=0.0, reward_hacking=True),
    ])

    result = evaluate_on_suite(
        str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
        task_provider=provider,
        episode_runner=lambda task, seed, path: next(outcomes),
    )

    assert provider.requested == ["held_a", "held_b"]
    assert result["heldout_pass_rate"] == 0.5
    assert result["reward_hacking_rate"] == 0.5
    assert result["reward_variance"] == 0.25
    record = json.loads((tmp_path / "runs/run-7/result.json").read_text())
    assert record["seed"] == 7
    assert record["config"]["heldout_envs"] == ["held_a", "held_b"]


def test_evaluate_refuses_train_heldout_leakage(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    _experiment(config_path, train=["train_a"], heldout=["leaked"])
    checkpoint_dir = tmp_path / "checkpoint"
    PolicyCheckpoint(
        objective="grpo", base_model="base", model_path="model", num_examples=2,
        mean_reward=0.5, train_envs=["leaked"], seed=7,
        experiment_config=_config_dict(train=["train_a"], heldout=["leaked"]),
    ).save(checkpoint_dir)

    with pytest.raises(ValueError, match="training environments do not match"):
        evaluate_on_suite(
            str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
            task_provider=_Provider(), episode_runner=lambda *_: None,
        )


def test_evaluate_rejects_env_without_compiled_tasks(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    _experiment(config_path, heldout=["missing"])
    checkpoint_dir = tmp_path / "checkpoint"
    PolicyCheckpoint(
        objective="grpo", base_model="base", model_path="model", num_examples=2,
        mean_reward=0.5, train_envs=["train_a"], seed=7,
        experiment_config=_config_dict(heldout=["missing"]),
    ).save(checkpoint_dir)

    class EmptyProvider:
        def tasks_for(self, domain, depth):
            return []

    with pytest.raises(ValueError, match="no compiled tasks"):
        evaluate_on_suite(
            str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
            task_provider=EmptyProvider(), episode_runner=lambda *_: None,
        )


def test_policy_adapter_uses_valid_tool_names_and_restores_endpoint():
    class Policy:
        def act(self, observation, action_types):
            assert action_types == frozenset({"action_0", "action_1"})
            assert observation["available_actions"][1]["policy_tool"] == "action_1"
            return {"type": "action_1", "message": "hello"}

    action = _PolicyContainerAdapter(Policy()).act(
        {"messages": []}, "send a message",
        [{"endpoint": "/draft"}, {"endpoint": "/send"}],
    )
    assert action == {"endpoint": "/send", "payload": {"message": "hello"}}


def test_reward_hacking_uses_existing_auditor_for_repeated_actions():
    repeated = SimpleNamespace(action={"endpoint": "/win"}, state_after={})
    result = SimpleNamespace(
        steps=[repeated, repeated, repeated, repeated],
        config=SimpleNamespace(objective="finish"),
    )
    assert _has_reward_hacking_pattern(result, passed=True) is True
    assert _has_reward_hacking_pattern(result, passed=False) is False
