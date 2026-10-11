"""Contracts for sandbox execution and microVM isolation.

Provides ports for container and hardware-isolated microVM runtimes (Firecracker).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class IsolationBackendType(str, Enum):
    """Supported sandbox isolation technologies."""

    FIRECRACKER = "firecracker"
    DOCKER = "docker"


@dataclass(frozen=True)
class SandboxInstance:
    """Active sandbox instance metadata."""

    instance_id: str
    backend_type: IsolationBackendType
    port: int
    env_name: str
    socket_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FirecrackerVMConfig:
    """Configuration for booting a Firecracker microVM instance."""

    vcpu_count: int = 2
    mem_size_mib: int = 1024
    kernel_image_path: str = "/tmp/forge-firecracker/vmlinux"
    rootfs_path: str = "/tmp/forge-firecracker/rootfs.ext4"
    boot_args: str = "console=ttyS0 reboot=k panic=1 pci=off init=/init root=/dev/vda rw"
    jailer_uid: int | None = 1000
    jailer_gid: int | None = 1000
    chroot_base: str = "/tmp/forge-jailer"
    net_host_dev: str = "tap0"
    guest_mac: str = "AA:FC:00:00:00:01"
    guest_ip: str = "172.16.0.2"
    host_ip: str = "172.16.0.1"
    netmask: str = "255.255.255.0"


@runtime_checkable
class SandboxIsolationBackend(Protocol):
    """Protocol for environment isolation providers (MicroVMs vs Containers)."""

    @property
    def backend_type(self) -> IsolationBackendType:
        """The active isolation technology."""
        ...

    def run(self, env_name: str, image_or_rootfs: str) -> tuple[str, int]:
        """Start an HTTP app sandbox and return (instance_id, bound_port)."""
        ...

    def run_cli(self, env_name: str) -> tuple[str, int]:
        """Start a sandboxed CLI shell environment."""
        ...

    def run_browser(self, env_name: str) -> tuple[str, int]:
        """Start an isolated browser environment with CDP relay."""
        ...

    def stop(self, instance_id: str) -> None:
        """Terminate and clean up a sandbox instance."""
        ...

    def is_healthy(self, instance_id: str) -> bool:
        """Check if the sandbox instance is active and responding."""
        ...
