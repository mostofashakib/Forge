"""Rootfs image and Copy-on-Write disk management for Firecracker microVMs.

Ensures that every agent episode runs with complete filesystem isolation,
preventing cross-episode state contamination or disk mutation of master templates.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class RootfsManager:
    """Manages disk images, copy-on-write clones, and lifecycle for microVM root filesystems."""

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        default_dir = os.getenv("FORGE_FIRECRACKER_DIR", "/tmp/forge-firecracker")
        self.storage_dir = Path(storage_dir or f"{default_dir}/images")

    def ensure_storage_dir(self) -> Path:
        """Create image storage directory if it does not exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        return self.storage_dir

    def clone_ephemeral_rootfs(
        self,
        base_rootfs: Path | str,
        vm_id: str,
        destination_dir: Path | str | None = None,
    ) -> Path:
        """Create a zero-contamination ephemeral copy of the base ext4 rootfs for a microVM instance.

        Uses sparse copy to preserve disk space and ensure sub-second rollout creation.
        """
        base_path = Path(base_rootfs)
        dest_dir = Path(destination_dir) if destination_dir else self.storage_dir / "active"
        dest_dir.mkdir(parents=True, exist_ok=True)
        target_path = dest_dir / f"{vm_id}-rootfs.ext4"

        if not base_path.exists():
            # If base rootfs does not exist on disk, create an empty sparse image for testing/init
            logger.warning("Base rootfs %s does not exist; creating sparse stub at %s", base_path, target_path)
            target_path.touch()
            return target_path

        # Try fast copy with copy-on-write / sparse flags
        try:
            # On Linux/macOS, cp with sparse preserves holes
            subprocess.run(
                ["cp", "--sparse=always", str(base_path), str(target_path)],
                check=True,
                capture_output=True,
            )
        except Exception:
            # Fallback to shutil.copyfile
            shutil.copyfile(base_path, target_path)

        # Set read-write permissions for the jailer/firecracker user
        try:
            os.chmod(target_path, 0o666)
        except Exception:
            pass

        logger.info("Created ephemeral microVM rootfs for %s at %s", vm_id, target_path)
        return target_path

    def cleanup_rootfs(self, target_path: Path | str) -> None:
        """Safely delete an ephemeral rootfs after microVM termination."""
        path = Path(target_path)
        if path.exists():
            try:
                path.unlink()
                logger.info("Cleaned up microVM rootfs %s", path)
            except Exception as exc:
                logger.warning("Failed to remove microVM rootfs %s: %exc", path, exc)
