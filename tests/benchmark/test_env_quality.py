import json
import pytest
from pathlib import Path
from forge.benchmark.env_quality import compute_env_quality, EnvQualityMetrics
from forge.schema.state_schema import StateSchemaManifest, FieldSpec


def _manifest() -> StateSchemaManifest:
    return StateSchemaManifest(
        env_name="email_env",
        fields={
            "inbox_count": FieldSpec(type="integer"),
            "selected_email": FieldSpec(type="object"),
        },
    )


def _write_episode(path: Path, steps: list[dict], summary: dict) -> None:
    lines = [json.dumps(s) for s in steps] + [json.dumps({"type": "episode_summary", **summary})]
    path.write_text("\n".join(lines))


def test_reward_density_all_nonzero(tmp_path):
    ep = tmp_path / "ep1.jsonl"
    _write_episode(ep, [
        {"step_index": 0, "reward": 0.5, "action": {"endpoint": "/star"}, "state_after": {"inbox_count": 3, "selected_email": {}}},
        {"step_index": 1, "reward": 0.8, "action": {"endpoint": "/reply"}, "state_after": {"inbox_count": 3, "selected_email": {}}},
    ], {"termination_reason": "success"})
    metrics = compute_env_quality(episode_dir=tmp_path, manifest=_manifest())
    assert metrics.reward_density == pytest.approx(1.0)


def test_reward_density_half_zero(tmp_path):
    ep = tmp_path / "ep1.jsonl"
    _write_episode(ep, [
        {"step_index": 0, "reward": 0.0, "action": {"endpoint": "/star"}, "state_after": {"inbox_count": 3}},
        {"step_index": 1, "reward": 0.5, "action": {"endpoint": "/reply"}, "state_after": {"inbox_count": 4}},
    ], {"termination_reason": "max_steps"})
    metrics = compute_env_quality(episode_dir=tmp_path, manifest=_manifest())
    assert metrics.reward_density == pytest.approx(0.5)


def test_dead_end_rate(tmp_path):
    for i, reason in enumerate(["dead_end", "success", "dead_end"]):
        ep = tmp_path / f"ep{i}.jsonl"
        _write_episode(ep, [
            {"step_index": 0, "reward": 0.0, "action": {"endpoint": "/a"}, "state_after": {}},
        ], {"termination_reason": reason})
    metrics = compute_env_quality(episode_dir=tmp_path, manifest=_manifest())
    assert metrics.dead_end_rate == pytest.approx(2 / 3)


def test_action_diversity(tmp_path):
    ep = tmp_path / "ep1.jsonl"
    _write_episode(ep, [
        {"step_index": 0, "reward": 0.1, "action": {"endpoint": "/star"}, "state_after": {}},
        {"step_index": 1, "reward": 0.1, "action": {"endpoint": "/star"}, "state_after": {}},
        {"step_index": 2, "reward": 0.1, "action": {"endpoint": "/reply"}, "state_after": {}},
    ], {"termination_reason": "max_steps"})
    metrics = compute_env_quality(episode_dir=tmp_path, manifest=_manifest())
    # 2 unique endpoints / 3 total = 0.667
    assert metrics.action_diversity == pytest.approx(2 / 3, rel=1e-2)


def test_state_coverage_score(tmp_path):
    ep = tmp_path / "ep1.jsonl"
    _write_episode(ep, [
        # Both fields present → coverage 1.0
        {"step_index": 0, "reward": 0.1, "action": {"endpoint": "/a"}, "state_after": {"inbox_count": 3, "selected_email": {}}},
        # Only one field → coverage 0.5
        {"step_index": 1, "reward": 0.1, "action": {"endpoint": "/b"}, "state_after": {"inbox_count": 4}},
    ], {"termination_reason": "max_steps"})
    metrics = compute_env_quality(episode_dir=tmp_path, manifest=_manifest())
    assert metrics.state_coverage_score == pytest.approx(0.75)


def test_returns_zero_metrics_on_empty_dir(tmp_path):
    metrics = compute_env_quality(episode_dir=tmp_path, manifest=_manifest())
    assert metrics.reward_density == 0.0
    assert metrics.dead_end_rate == 0.0
