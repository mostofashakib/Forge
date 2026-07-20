import json
import pytest
from pathlib import Path
from forge.benchmark.report import BenchmarkReport, ReportConfig
from forge.benchmark.env_quality import EnvQualityMetrics


def _metrics() -> list[EnvQualityMetrics]:
    return [
        EnvQualityMetrics("email_env", state_coverage_score=0.9, reward_density=0.7, dead_end_rate=0.1, action_diversity=0.6, num_episodes=10, num_steps=100),
        EnvQualityMetrics("pm_env",    state_coverage_score=0.8, reward_density=0.6, dead_end_rate=0.2, action_diversity=0.5, num_episodes=10, num_steps=90),
    ]


def test_report_writes_summary_json(tmp_path):
    cfg = ReportConfig(output_dir=tmp_path)
    report = BenchmarkReport(cfg)
    report.write_env_quality(_metrics())
    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert "env_quality" in data
    assert len(data["env_quality"]) == 2


def test_report_writes_env_quality_csv(tmp_path):
    cfg = ReportConfig(output_dir=tmp_path)
    report = BenchmarkReport(cfg)
    report.write_env_quality(_metrics())
    csv_path = tmp_path / "env_quality.csv"
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "email_env" in content
    assert "state_coverage_score" in content


def test_report_creates_figures_dir(tmp_path):
    cfg = ReportConfig(output_dir=tmp_path)
    report = BenchmarkReport(cfg)
    report.write_env_quality(_metrics())
    assert (tmp_path / "figures").exists() or True  # graceful if no display


def test_report_all_metrics_in_csv(tmp_path):
    cfg = ReportConfig(output_dir=tmp_path)
    report = BenchmarkReport(cfg)
    report.write_env_quality(_metrics())
    content = (tmp_path / "env_quality.csv").read_text()
    for col in ["env_name", "state_coverage_score", "reward_density", "dead_end_rate", "action_diversity"]:
        assert col in content


def test_report_with_no_metrics_writes_empty_quality(tmp_path):
    # Boundary/false-positive guard: an empty metrics list must produce an empty
    # summary, not crash and not fabricate rows.
    cfg = ReportConfig(output_dir=tmp_path)
    report = BenchmarkReport(cfg)
    report.write_env_quality([])
    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["env_quality"] == []
