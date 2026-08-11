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
from forge.grading_provenance import GraderContaminationError
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
        + "reward_preset: full_layered_partial\nbase_model: base\n"
        + f"seeds: {seeds or [7]}\n",
        encoding="utf-8",
    )


def _config_dict(train=None, heldout=None, seeds=None):
    return {
        "train_envs": train or ["train_a"],
        "heldout_envs": heldout or ["held_a", "held_b"],
        "reward_preset": "full_layered_partial",
        "base_model": "base",
        "seeds": seeds or [7],
        "determinism_repeats": 2,
        "require_grader_independence": True,
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
        EpisodeOutcome(passed=True, reward=0.5, reward_hacking=False),
        # The full preset's auditor turns this apparent pass into zero reward.
        EpisodeOutcome(passed=True, reward=0.9, reward_hacking=True),
        EpisodeOutcome(passed=False, reward=0.0, reward_hacking=False),
    ])

    result = evaluate_on_suite(
        str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
        task_provider=provider,
        episode_runner=lambda task, seed, path: next(outcomes),
    )

    assert provider.requested == ["held_a", "held_b"]
    assert result["heldout_pass_rate"] == 0.5
    assert result["reward_hacking_rate"] == 0.25
    assert result["reward_variance"] == 0.03125
    assert result["num_eval_tasks"] == 2
    assert result["num_eval_episodes"] == 4
    record = json.loads((tmp_path / "runs/run-7/result.json").read_text())
    assert record["seed"] == 7
    assert record["determinism"] == "on"
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


def test_full_no_auditor_records_hacking_without_zeroing_reward(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DETERMINISM", "off")
    config_path = tmp_path / "experiment.yaml"
    _experiment(config_path, heldout=["held_a"])
    text = config_path.read_text().replace(
        "full_layered_partial", "full_no_auditor"
    )
    config_path.write_text(text)
    config = _config_dict(heldout=["held_a"])
    config["reward_preset"] = "full_no_auditor"
    checkpoint_dir = tmp_path / "checkpoint"
    PolicyCheckpoint(
        objective="grpo", base_model="base", model_path="model", num_examples=2,
        mean_reward=0.5, train_envs=["train_a"], seed=7,
        experiment_config=config,
    ).save(checkpoint_dir)

    result = evaluate_on_suite(
        str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
        task_provider=_Provider(),
        episode_runner=lambda *_: EpisodeOutcome(True, 0.8, True),
    )
    assert result["heldout_pass_rate"] == 1.0
    assert result["reward_hacking_rate"] == 1.0
    assert result["determinism"] == "off"


# ---------------------------------------------------------------------------
# Generator/grader independence
# ---------------------------------------------------------------------------

def _judged_experiment(path, *, independence=True):
    path.write_text(
        "train_envs: [train_a]\nheldout_envs: [held_a]\n"
        "reward_preset: judge_only\nbase_model: base\nseeds: [7]\n"
        f"require_grader_independence: {'true' if independence else 'false'}\n",
        encoding="utf-8",
    )


def _judged_config(independence=True):
    return {
        "train_envs": ["train_a"], "heldout_envs": ["held_a"],
        "reward_preset": "judge_only", "base_model": "base", "seeds": [7],
        "determinism_repeats": 2, "require_grader_independence": independence,
    }


def _checkpoint(tmp_path, config):
    checkpoint_dir = tmp_path / "checkpoint"
    PolicyCheckpoint(
        objective="grpo", base_model="base", model_path="model", num_examples=2,
        mean_reward=0.5, train_envs=["train_a"], seed=7, run_id="run-7",
        experiment_config=config,
    ).save(checkpoint_dir)
    return checkpoint_dir


def _pass_runner(task, seed, path):
    return EpisodeOutcome(passed=True, reward=1.0, reward_hacking=False)


def test_structural_run_records_independent_grading_provenance(tmp_path):
    """The default preset issues no LLM verdict, so provenance is clean."""
    config_path = tmp_path / "experiment.yaml"
    _experiment(config_path, heldout=["held_a"])
    checkpoint_dir = _checkpoint(tmp_path, _config_dict(heldout=["held_a"]))

    result = evaluate_on_suite(
        str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
        task_provider=_Provider(), episode_runner=_pass_runner,
    )

    grading = result["grading"]
    assert grading["llm_graded"] is False
    assert grading["independent"] is True
    assert grading["judge_model"] is None


def test_llm_graded_run_refuses_a_judge_from_the_generating_family(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "claude-sonnet-4-6")
    config_path = tmp_path / "experiment.yaml"
    _judged_experiment(config_path)
    checkpoint_dir = _checkpoint(tmp_path, _judged_config())

    with pytest.raises(GraderContaminationError):
        evaluate_on_suite(
            str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
            task_provider=_Provider(), episode_runner=_pass_runner,
        )


def test_llm_graded_run_accepts_a_judge_from_another_family(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "gpt-4o")
    config_path = tmp_path / "experiment.yaml"
    _judged_experiment(config_path)
    checkpoint_dir = _checkpoint(tmp_path, _judged_config())

    result = evaluate_on_suite(
        str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
        task_provider=_Provider(), episode_runner=_pass_runner,
    )

    assert result["grading"]["independent"] is True
    assert result["grading"]["judge_family"] == "gpt"


def test_waiving_independence_records_the_contamination_instead_of_aborting(tmp_path, monkeypatch):
    """An explicit waiver must still leave the contamination visible in the record."""
    monkeypatch.setenv("FORGE_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "claude-sonnet-4-6")
    config_path = tmp_path / "experiment.yaml"
    _judged_experiment(config_path, independence=False)
    checkpoint_dir = _checkpoint(tmp_path, _judged_config(independence=False))

    result = evaluate_on_suite(
        str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
        task_provider=_Provider(), episode_runner=_pass_runner,
    )

    record = json.loads((tmp_path / "runs/run-7/result.json").read_text())
    assert result["grading"]["independent"] is False
    assert record["grading"]["independent"] is False


def test_structural_run_is_not_blocked_by_a_same_family_judge_variable(tmp_path, monkeypatch):
    """False-positive guard: a configured judge that never grades is not contamination."""
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "claude-sonnet-4-6")
    config_path = tmp_path / "experiment.yaml"
    _experiment(config_path, heldout=["held_a"])
    checkpoint_dir = _checkpoint(tmp_path, _config_dict(heldout=["held_a"]))

    result = evaluate_on_suite(
        str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
        task_provider=_Provider(), episode_runner=_pass_runner,
    )

    assert result["grading"]["independent"] is True


# ---------------------------------------------------------------------------
# Observed grading — the record must reflect what ran, not what the preset implies
# ---------------------------------------------------------------------------

def test_result_records_the_observed_verdict_count(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "gpt-4o")
    config_path = tmp_path / "experiment.yaml"
    _judged_experiment(config_path)
    checkpoint_dir = _checkpoint(tmp_path, _judged_config())

    result = evaluate_on_suite(
        str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
        task_provider=_Provider(),
        episode_runner=lambda task, seed, path: EpisodeOutcome(
            passed=True, reward=1.0, reward_hacking=False, llm_verdicts=5
        ),
    )

    record = json.loads((tmp_path / "runs/run-7/result.json").read_text())
    assert result["grading"]["llm_verdicts"] == 10  # 2 repeats x 5 verdicts
    assert record["grading"]["llm_verdicts"] == 10


def test_structural_run_that_issues_verdicts_is_refused_not_recorded(tmp_path):
    """A runner that under-declares itself must abort, not write a false record."""
    config_path = tmp_path / "experiment.yaml"
    _experiment(config_path, heldout=["held_a"])
    checkpoint_dir = _checkpoint(tmp_path, _config_dict(heldout=["held_a"]))

    with pytest.raises(ValueError, match="under-declared"):
        evaluate_on_suite(
            str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
            task_provider=_Provider(),
            episode_runner=lambda task, seed, path: EpisodeOutcome(
                passed=True, reward=1.0, reward_hacking=False, llm_verdicts=3
            ),
        )
    assert not (tmp_path / "runs/run-7/result.json").exists()


def test_structural_run_observing_no_verdicts_records_zero(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    _experiment(config_path, heldout=["held_a"])
    checkpoint_dir = _checkpoint(tmp_path, _config_dict(heldout=["held_a"]))

    result = evaluate_on_suite(
        str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
        task_provider=_Provider(), episode_runner=_pass_runner,
    )

    assert result["grading"]["llm_graded"] is False
    assert result["grading"]["llm_verdicts"] == 0


def test_a_runner_declaring_llm_verdicts_is_gated_before_any_episode(tmp_path, monkeypatch):
    """The container runner always scores with an LLM, so the gate must see that.

    The preset alone says ``full_layered_partial`` issues no LLM verdict, so a
    preset-derived flag would let this contaminated run proceed.
    """
    monkeypatch.setenv("FORGE_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("FORGE_LLM_MODEL_CAPABLE", "claude-sonnet-4-6")
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "claude-sonnet-4-6")
    config_path = tmp_path / "experiment.yaml"
    _experiment(config_path, heldout=["held_a"])
    checkpoint_dir = _checkpoint(tmp_path, _config_dict(heldout=["held_a"]))

    episodes_run = []

    def declaring_runner(task, seed, path):
        episodes_run.append(task)
        return EpisodeOutcome(passed=True, reward=1.0, reward_hacking=False)

    declaring_runner.issues_llm_verdicts = True

    with pytest.raises(GraderContaminationError):
        evaluate_on_suite(
            str(checkpoint_dir), str(config_path), runs_dir=tmp_path / "runs",
            task_provider=_Provider(), episode_runner=declaring_runner,
        )
    assert episodes_run == []


def test_container_episode_runner_declares_that_it_issues_llm_verdicts(tmp_path, monkeypatch):
    """Every container episode is scored by ObjectiveScorer, so it grades with an LLM."""
    from forge.benchmark._eval import _container_episode_runner

    import forge.benchmark._eval as eval_module

    monkeypatch.setattr(eval_module, "load_policy_agent", lambda _dir: object())
    checkpoint_dir = _checkpoint(tmp_path, _config_dict())
    runner = _container_episode_runner(checkpoint_dir, "full_layered_partial")

    assert runner.issues_llm_verdicts is True
