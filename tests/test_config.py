"""Tests for centralized Forge configuration file and version settings."""
from __future__ import annotations

from pathlib import Path

from forge.config import (
    EvaluationConfig,
    ForgeConfig,
    SandboxConfig,
    StorageConfig,
    VersionConfig,
    get_config,
    reset_config,
    set_config,
)
from forge.settings import (
    database_url,
    determinism_enabled,
    generated_envs_root,
    redis_url,
    sandbox_limit,
)


def test_centralized_config_defaults():
    cfg = get_config()

    # Version numbers
    assert cfg.versions.forge_version == "0.3.0"
    assert cfg.versions.contract_version == "1.0.0"
    assert cfg.versions.default_prompt_version == "v1.0.0"
    assert cfg.versions.default_verifier_version == "v1.0.0"
    assert cfg.versions.default_data_version == "v1.0.0"
    assert cfg.versions.default_test_suite_version == "v1.0.0"
    assert cfg.versions.cli_base_version == "ubuntu-22.04"

    # Sandbox
    assert cfg.sandbox.runtime_backend in ("auto", "firecracker", "docker")
    assert cfg.sandbox.vcpu_count == 2
    assert cfg.sandbox.mem_size_mib == 1024
    assert cfg.sandbox.app_port == 8000
    assert cfg.sandbox.cdp_port == 9222

    # Evaluation
    assert cfg.evaluation.max_steps == 50
    assert cfg.evaluation.http_timeout_seconds == 10.0
    assert cfg.evaluation.success_threshold == 1.0

    # Storage
    assert "sqlite" in cfg.storage.database_url
    assert "redis" in cfg.storage.redis_url


def test_load_from_custom_yaml_file(tmp_path: Path):
    custom_yaml = tmp_path / "custom_forge.yaml"
    custom_yaml.write_text(
        """
versions:
  forge_version: "2.0.0-custom"
  default_prompt_version: "v3.1.4"
  default_verifier_version: "v2.5.0"
sandbox:
  vcpu_count: 8
  mem_size_mib: 4096
  runtime_backend: "firecracker"
evaluation:
  max_steps: 120
storage:
  database_url: "sqlite:///./custom.db"
"""
    )

    loaded = ForgeConfig.load(custom_yaml)
    assert loaded.versions.forge_version == "2.0.0-custom"
    assert loaded.versions.default_prompt_version == "v3.1.4"
    assert loaded.versions.default_verifier_version == "v2.5.0"
    assert loaded.sandbox.vcpu_count == 8
    assert loaded.sandbox.mem_size_mib == 4096
    assert loaded.sandbox.runtime_backend == "firecracker"
    assert loaded.evaluation.max_steps == 120
    assert loaded.storage.database_url == "sqlite:///./custom.db"


def test_settings_delegation_to_central_config():
    custom = ForgeConfig(
        storage=StorageConfig(
            database_url="sqlite:///./mock_test.db",
            redis_url="redis://custom-redis:6379/1",
            generated_envs_dir="custom_envs",
        ),
        sandbox=SandboxConfig(sandbox_limit=42),
    )
    set_config(custom)
    try:
        assert database_url() == "sqlite:///./mock_test.db"
        assert redis_url() == "redis://custom-redis:6379/1"
        assert str(generated_envs_root()) == "custom_envs"
        assert sandbox_limit() == 42
    finally:
        reset_config()


def test_corrupt_yaml_file_falls_back_to_defaults(tmp_path: Path):
    corrupt_yaml = tmp_path / "corrupt.yaml"
    corrupt_yaml.write_text(":::invalid: [yaml: unparseable {[[")

    loaded = ForgeConfig.load(corrupt_yaml)
    # Negative case: corrupt file must NOT crash and must fall back to valid defaults
    assert loaded.versions.forge_version == "0.3.0"
    assert loaded.sandbox.app_port == 8000
    assert loaded.storage.database_url != "corrupt"


def test_invalid_determinism_value_raises(monkeypatch):
    import pytest

    monkeypatch.setenv("FORGE_DETERMINISM", "invalid_mode")
    # False-positive / negative case: look-valid but unsupported mode must be rejected
    with pytest.raises(ValueError, match="must be 'on' or 'off'"):
        determinism_enabled()


def test_nonexistent_config_file_returns_clean_defaults(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.yaml"
    assert not missing_path.exists()

    loaded = ForgeConfig.load(missing_path)
    assert loaded.versions.forge_version is not None
    assert loaded.sandbox.vcpu_count == 2

