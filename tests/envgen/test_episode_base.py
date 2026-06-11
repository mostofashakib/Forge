# tests/envgen/test_episode_base.py
import json
from forge.envgen.episode_base import (
    BaseEpisodeConfig,
    BaseEpisodeResult,
    TerminationMonitor,
)


def monitor(**overrides) -> TerminationMonitor:
    cfg = BaseEpisodeConfig(objective="test", **overrides)
    return TerminationMonitor(cfg)


# ---------------------------------------------------------------------------
# TerminationMonitor — the shared early-stop logic for every env type
# ---------------------------------------------------------------------------

def test_success_threshold_terminates():
    m = monitor(success_threshold=0.9)
    assert m.observe(0.5) is None
    assert m.observe(0.95) == "success"


def test_identical_score_plateau_is_dead_end():
    m = monitor(dead_end_patience=3, success_threshold=0.99)
    assert m.observe(0.5) is None
    assert m.observe(0.5) is None
    assert m.observe(0.5) == "dead_end"


def test_state_hash_markers_drive_dead_end_for_container_envs():
    m = monitor(dead_end_patience=3, success_threshold=0.99)
    # scores fluctuate but state never changes
    assert m.observe(0.50, marker="hash_a") is None
    assert m.observe(0.61, marker="hash_a") is None
    assert m.observe(0.55, marker="hash_a") == "dead_end"


def test_changing_markers_are_not_dead_end():
    m = monitor(dead_end_patience=3, success_threshold=0.99)
    for i, score in enumerate([0.5, 0.5, 0.5, 0.5]):
        assert m.observe(score, marker=f"hash_{i}") is None


def test_consecutive_low_scores_diverge():
    m = monitor(divergence_threshold=0.3, consecutive_below_threshold=3, dead_end_patience=99)
    assert m.observe(0.1, marker="a") is None
    assert m.observe(0.15, marker="b") is None
    assert m.observe(0.05, marker="c") == "diverged"


def test_recovering_score_resets_divergence_counter():
    m = monitor(divergence_threshold=0.3, consecutive_below_threshold=3, dead_end_patience=99)
    m.observe(0.1, marker="a")
    m.observe(0.1, marker="b")
    m.observe(0.6, marker="c")  # recovery
    assert m.observe(0.1, marker="d") is None


def test_success_wins_over_dead_end():
    m = monitor(success_threshold=0.9, dead_end_patience=2)
    m.observe(0.95, marker="same")  # success on first qualifying score
    m2 = monitor(success_threshold=0.9, dead_end_patience=2)
    m2.observe(0.95, marker="same")
    assert m2.observe(0.95, marker="same") == "success"


# ---------------------------------------------------------------------------
# BaseEpisodeResult — shared recording + JSONL export
# ---------------------------------------------------------------------------

def test_result_jsonl_has_step_lines_and_summary():
    result = BaseEpisodeResult()
    result.steps.append({"step_index": 0, "reward": 0.5})
    result.steps.append({"step_index": 1, "reward": 0.7})
    result.total_reward = 1.2
    result.termination_reason = "success"

    lines = result.to_jsonl().strip().split("\n")
    assert len(lines) == 3
    assert json.loads(lines[0])["step_index"] == 0
    summary = json.loads(lines[-1])
    assert summary["type"] == "episode_summary"
    assert summary["total_steps"] == 2
    assert summary["total_reward"] == 1.2
    assert summary["termination_reason"] == "success"
    assert "started_at" in summary


def test_result_write_jsonl_creates_parent_dirs(tmp_path):
    result = BaseEpisodeResult()
    result.steps.append({"step_index": 0})
    target = tmp_path / "episodes" / "nested" / "ep.jsonl"
    result.write_jsonl(target)
    assert target.exists()
    assert json.loads(target.read_text().strip().split("\n")[-1])["type"] == "episode_summary"


# ---------------------------------------------------------------------------
# Runners actually build on the shared base
# ---------------------------------------------------------------------------

def test_all_runner_configs_extend_the_base():
    from forge.envgen.browser_runner import BrowserEpisodeConfig
    from forge.envgen.cli_runner import CliEpisodeConfig
    from forge.envgen.episode_runner import EpisodeConfig

    for config_type in (CliEpisodeConfig, BrowserEpisodeConfig, EpisodeConfig):
        assert issubclass(config_type, BaseEpisodeConfig)


def test_all_runner_results_extend_the_base():
    from forge.envgen.browser_runner import BrowserEpisodeResult
    from forge.envgen.cli_runner import CliEpisodeResult
    from forge.envgen.episode_runner import EpisodeResult

    for result_type in (CliEpisodeResult, BrowserEpisodeResult, EpisodeResult):
        assert issubclass(result_type, BaseEpisodeResult)
