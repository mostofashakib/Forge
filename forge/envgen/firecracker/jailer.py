"""Firecracker Jailer process supervisor and security isolation.

The AWS Jailer creates an isolated chroot boundary, sets up dedicated cgroups,
drops privileges to an unprivileged UID/GID, and execs the Firecracker binary.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class JailerConfig:
    """Configuration for running Firecracker inside an AWS Jailer environment."""

    vm_id: str
    exec_file: str = "firecracker"
    uid: int = 1000
    gid: int = 1000
    chroot_base_dir: str = "/srv/jailer"
    netns: str | None = None
    daemonize: bool = False


class FirecrackerJailer:
    """Supervises Firecracker microVM processes either jailed or direct."""

    def __init__(
        self,
        jailer_bin: str | None = None,
        firecracker_bin: str | None = None,
    ) -> None:
        self.jailer_bin = jailer_bin or shutil.which("jailer") or "/usr/bin/jailer"
        self.firecracker_bin = firecracker_bin or shutil.which("firecracker") or "/usr/bin/firecracker"

    def jail_chroot_dir(self, config: JailerConfig) -> Path:
        """Return the chroot root directory created by the jailer for this VM."""
        binary_name = Path(config.exec_file).name
        return Path(config.chroot_base_dir) / binary_name / config.vm_id / "root"

    def build_jailer_command(self, config: JailerConfig, socket_path: str = "run/firecracker.socket") -> list[str]:
        """Construct the CLI arguments for invoking the jailer."""
        cmd = [
            self.jailer_bin,
            "--id", config.vm_id,
            "--exec-file", self.firecracker_bin,
            "--uid", str(config.uid),
            "--gid", str(config.gid),
            "--chroot-base-dir", config.chroot_base_dir,
        ]
        if config.netns:
            cmd.extend(["--netns", config.netns])
        if config.daemonize:
            cmd.append("--daemonize")

        # Arguments passed to the internal Firecracker binary after '--'
        cmd.extend(["--", "--api-sock", socket_path])
        return cmd

    def build_direct_command(self, socket_path: Path | str) -> list[str]:
        """Construct direct Firecracker invocation (used when running unjailed)."""
        return [
            self.firecracker_bin,
            "--api-sock", str(socket_path),
        ]

    def spawn(
        self,
        vm_id: str,
        socket_path: Path | str,
        use_jailer: bool = False,
        config: JailerConfig | None = None,
    ) -> subprocess.Popen:
        """Spawn a Firecracker microVM process."""
        if use_jailer and os.path.exists(self.jailer_bin):
            cfg = config or JailerConfig(vm_id=vm_id)
            cmd = self.build_jailer_command(cfg, socket_path=str(socket_path))
            logger.info("Spawning jailed Firecracker microVM %s: %s", vm_id, " ".join(cmd))
        else:
            cmd = self.build_direct_command(socket_path)
            logger.info("Spawning direct Firecracker microVM %s: %s", vm_id, " ".join(cmd))

        # Ensure parent directory of socket exists
        sock_p = Path(socket_path)
        sock_p.parent.mkdir(parents=True, exist_ok=True)
        if sock_p.exists():
            sock_p.unlink()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return process
