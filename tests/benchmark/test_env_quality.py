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


def test_quality_metrics_stream_large_files_without_retaining_endpoint_history(tmp_path):
    import tracemalloc

    path = tmp_path / "large.jsonl"
    record = json.dumps({"reward": 0.5, "action": {"endpoint": "/" + "a" * 128}, "state_after": {}})
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(20000):
            handle.write(record + "\n")

    tracemalloc.start()
    try:
        metrics = compute_env_quality(tmp_path, _manifest())
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert metrics.num_steps == 20000
    assert metrics.num_episodes == 1
    assert metrics.action_diversity == pytest.approx(1 / 20000)
    assert peak < 1024 * 1024


@pytest.mark.parametrize("bad_record", ["{broken", "null", "[]"])
def test_quality_metrics_report_bad_records_and_continue_reading_partial_traces(tmp_path, caplog, bad_record):
    path = tmp_path / "partial.jsonl"
    path.write_text(
        '{"reward": 0.0}\n' + bad_record + '\n{"reward": 1.0}\n', encoding="utf-8",
    )
    metrics = compute_env_quality(tmp_path, _manifest())
    assert metrics.num_steps == 2
    assert metrics.num_episodes == 1
    assert metrics.reward_density == 0.5
    assert "partial.jsonl" in caplog.text


def test_partial_trace_with_invalid_utf8_still_counts_its_retained_steps_as_an_episode(tmp_path, caplog):
    path = tmp_path / "encoding_error.jsonl"
    path.write_bytes(b'{"reward": 1.0}\n' * 1000 + b'\xff')

    metrics = compute_env_quality(tmp_path, _manifest())

    assert metrics.num_steps > 0
    assert metrics.num_episodes == 1
    assert metrics.reward_density == 1.0
    assert "encoding_error.jsonl" in caplog.text
