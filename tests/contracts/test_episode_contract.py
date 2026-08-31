from __future__ import annotations

from pathlib import Path

import pytest

from forge.contracts import BaseEpisodeConfig, BaseEpisodeResult, EpisodeController


class _OneStep(EpisodeController):
    def run_episode(
        self,
        agent,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        jsonl_path: Path | None = None,
    ) -> BaseEpisodeResult:
        result = BaseEpisodeResult()
        result.termination_reason = f"seed={seed}"
        return result


def test_a_controller_runs_an_episode_and_returns_a_result():
    result = _OneStep().run_episode(agent=None, seed=7)
    assert result.termination_reason == "seed=7"


def test_seed_is_uniform_across_controllers_even_when_unused():
    # False-positive guard: a family with no seeding path still accepts the
    # keyword, so callers need no per-family special case.
    assert _OneStep().run_episode(agent=None).termination_reason == "seed=None"


def test_a_controller_missing_run_episode_cannot_be_instantiated():
    class Incomplete(EpisodeController):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_episode_config_defaults_match_the_documented_thresholds():
    config = BaseEpisodeConfig(objective="close the ticket")
    assert config.max_steps == 30
    assert config.divergence_threshold == 0.2
    assert config.consecutive_below_threshold == 3
    assert config.dead_end_patience == 5
    assert config.success_threshold == 0.9


def test_episode_results_convert_to_the_shared_rollout_contract():
    result = BaseEpisodeResult(
        steps=[{
            "action": {"type": "close"},
            "reward": 1.0,
            "terminated": True,
            "truncated": False,
        }],
        total_reward=1.0,
        termination_reason="success",
    )

    record = result.to_rollout_record(
        env_name="support", task_name="close-ticket", seed=7
    )

    assert record.outcome == "success"
    assert record.per_step_rewards == [1.0]
    assert record.steps == 1
    assert '\"type\": \"close\"' in record.completion


def test_rollout_success_comes_from_verdict_not_termination_reason():
    result = BaseEpisodeResult(
        total_reward=1.0,
        termination_reason="submitted",
        passed=True,
    )

    record = result.to_rollout_record(task_name="done")

    assert record.passed is True
    assert record.outcome == "success"
    assert record.termination_reason == "submitted"
