"""Centralized Configuration for Forge.

Houses all version numbers, runtime parameters, sandbox infrastructure limits,
timeouts, and storage endpoints in a single authoritative, configurable location.
Can be overridden via environment variables or a YAML/JSON config file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VersionConfig:
    """Centralized version numbers across prompts, verifiers, data, and system contracts."""

    forge_version: str = "0.3.0"
    contract_version: str = "1.0.0"
    default_prompt_version: str = "v1.0.0"
    default_verifier_version: str = "v1.0.0"
    default_data_version: str = "v1.0.0"
    default_test_suite_version: str = "v1.0.0"
    cli_base_version: str = "ubuntu-22.04"

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> VersionConfig:
        o = overrides or {}
        return cls(
            forge_version=o.get("forge_version", os.environ.get("FORGE_VERSION", "0.3.0")),
            contract_version=o.get("contract_version", os.environ.get("FORGE_CONTRACT_VERSION", "1.0.0")),
            default_prompt_version=o.get("default_prompt_version", os.environ.get("FORGE_PROMPT_VERSION", "v1.0.0")),
            default_verifier_version=o.get("default_verifier_version", os.environ.get("FORGE_VERIFIER_VERSION", "v1.0.0")),
            default_data_version=o.get("default_data_version", os.environ.get("FORGE_DATA_VERSION", "v1.0.0")),
            default_test_suite_version=o.get("default_test_suite_version", os.environ.get("FORGE_TEST_SUITE_VERSION", "v1.0.0")),
            cli_base_version=o.get("cli_base_version", os.environ.get("FORGE_CLI_BASE_VERSION", "ubuntu-22.04")),
        )


@dataclass(frozen=True)
class SandboxConfig:
    """Sandbox virtualization and container infrastructure settings."""

    runtime_backend: str = "auto"
    firecracker_dir: str = "/tmp/forge-firecracker"
    jailer_dir: str = "/tmp/forge-jailer"
    vcpu_count: int = 2
    mem_size_mib: int = 1024
    kernel_image_path: str = "/tmp/forge-firecracker/vmlinux"
    rootfs_path: str = "/tmp/forge-firecracker/rootfs.ext4"
    jailer_uid: int = 1000
    jailer_gid: int = 1000
    sandbox_limit: int = 10
    app_port: int = 8000
    cdp_port: int = 9222

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> SandboxConfig:
        o = overrides or {}
        fc_dir = o.get("firecracker_dir", os.environ.get("FORGE_FIRECRACKER_DIR", "/tmp/forge-firecracker"))
        return cls(
            runtime_backend=o.get("runtime_backend", os.environ.get("FORGE_SANDBOX_RUNTIME", "auto")),
            firecracker_dir=fc_dir,
            jailer_dir=o.get("jailer_dir", os.environ.get("FORGE_JAILER_DIR", "/tmp/forge-jailer")),
            vcpu_count=int(o.get("vcpu_count", os.environ.get("FORGE_VCPU_COUNT", 2))),
            mem_size_mib=int(o.get("mem_size_mib", os.environ.get("FORGE_MEM_SIZE_MIB", 1024))),
            kernel_image_path=o.get("kernel_image_path", os.environ.get("FORGE_KERNEL_IMAGE", f"{fc_dir}/vmlinux")),
            rootfs_path=o.get("rootfs_path", os.environ.get("FORGE_ROOTFS_IMAGE", f"{fc_dir}/rootfs.ext4")),
            jailer_uid=int(o.get("jailer_uid", os.environ.get("FORGE_JAILER_UID", 1000))),
            jailer_gid=int(o.get("jailer_gid", os.environ.get("FORGE_JAILER_GID", 1000))),
            sandbox_limit=int(o.get("sandbox_limit", os.environ.get("FORGE_SANDBOX_LIMIT", 10))),
            app_port=int(o.get("app_port", os.environ.get("FORGE_APP_PORT", 8000))),
            cdp_port=int(o.get("cdp_port", os.environ.get("FORGE_CDP_PORT", 9222))),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation, stepping limits, and experiment determinism."""

    max_steps: int = 50
    http_timeout_seconds: float = 10.0
    success_threshold: float = 1.0
    dead_end_patience: int = 8
    experiment_seed: int = 42
    determinism_enabled: bool = True

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> EvaluationConfig:
        o = overrides or {}
        det_raw = os.environ.get("FORGE_DETERMINISM", "on").strip().lower()
        return cls(
            max_steps=int(o.get("max_steps", os.environ.get("FORGE_MAX_STEPS", 50))),
            http_timeout_seconds=float(o.get("http_timeout_seconds", os.environ.get("FORGE_HTTP_TIMEOUT", 10.0))),
            success_threshold=float(o.get("success_threshold", os.environ.get("FORGE_SUCCESS_THRESHOLD", 1.0))),
            dead_end_patience=int(o.get("dead_end_patience", os.environ.get("FORGE_DEAD_END_PATIENCE", 8))),
            experiment_seed=int(o.get("experiment_seed", os.environ.get("FORGE_EXPERIMENT_SEED", 42))),
            determinism_enabled=o.get("determinism_enabled", det_raw == "on"),
        )


@dataclass(frozen=True)
class StorageConfig:
    """Database, cache, and filesystem roots."""

    database_url: str = "sqlite:///./forge.db"
    redis_url: str = "redis://localhost:6379/0"
    generated_envs_dir: str = "generated_envs"

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> StorageConfig:
        o = overrides or {}
        return cls(
            database_url=o.get("database_url", os.environ.get("FORGE_DB_URL", "sqlite:///./forge.db")),
            redis_url=o.get("redis_url", os.environ.get("REDIS_URL", "redis://localhost:6379/0")),
            generated_envs_dir=o.get("generated_envs_dir", os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs")),
        )


@dataclass(frozen=True)
class ForgeConfig:
    """Unified configuration tree for Forge."""

    versions: VersionConfig = field(default_factory=VersionConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> ForgeConfig:
        """Load configuration from a file (if present) and merge with environment variables."""
        file_path = config_path or os.environ.get("FORGE_CONFIG_FILE", "forge.config.yaml")
        file_data: dict[str, Any] = {}

        p = Path(file_path)
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8")
                parsed = yaml.safe_load(content)
                if isinstance(parsed, dict):
                    file_data = parsed
            except Exception:
                pass

        return cls(
            versions=VersionConfig.from_env(file_data.get("versions")),
            sandbox=SandboxConfig.from_env(file_data.get("sandbox")),
            evaluation=EvaluationConfig.from_env(file_data.get("evaluation")),
            storage=StorageConfig.from_env(file_data.get("storage")),
        )


# Global singleton holder
_ACTIVE_CONFIG: ForgeConfig | None = None


def get_config() -> ForgeConfig:
    """Return the active global configuration."""
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        _ACTIVE_CONFIG = ForgeConfig.load()
    return _ACTIVE_CONFIG


def set_config(config: ForgeConfig) -> None:
    """Explicitly set or override the active configuration."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config


def reset_config() -> None:
    """Reset configuration to re-read from file and environment."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = None
