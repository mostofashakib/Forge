"""Identify the exact environment an episode ran against."""
from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from forge.settings import generated_envs_root

logger = logging.getLogger("forge.reliability")


def compute_environment_version(
    env_name: str,
    env_type: str = "general",
    container_id: str | None = None,
    image_id: str | None = None,
    env_dir: Path | str | None = None,
) -> str:
    """Return Docker image ID for container environments, or hash of package for in-process ones."""
    if image_id:
        return f"docker:{image_id}"

    if env_type in ("general", "premade", "cli", "browser") and container_id:
        try:
            import docker
            client = docker.from_env()
            try:
                c = client.containers.get(container_id)
                return f"docker:{c.image.id}"
            finally:
                client.close()
        except Exception:
            pass

    target_dir = Path(env_dir) if env_dir is not None else (generated_envs_root() / env_name)
    if target_dir.exists():
        hasher = hashlib.sha256()
        try:
            for rel, path in _package_files(target_dir):
                hasher.update(rel.encode("utf-8"))
                hasher.update(path.read_bytes())
            return f"package:{hasher.hexdigest()[:16]}"
        except Exception as exc:
            logger.debug("Failed computing package hash for %s: %s", env_name, exc)

    if env_type in ("general", "premade", "cli", "browser"):
        return f"docker:env_{env_name}:{env_type}"

    return f"package:hash_{hashlib.sha256(env_name.encode()).hexdigest()[:12]}"


# Runs write these beside the package. They are outputs, not the environment,
# so they never change its version and are never walked.
_RUN_OUTPUT_DIRS = frozenset({"episodes", "agent_episodes", "exports", "__pycache__"})
_RUN_OUTPUT_FILES = frozenset({"port"})


def _package_files(root: Path) -> Iterator[tuple[str, Path]]:
    """(relative posix path, path) for every file that defines the environment, sorted."""
    found: list[tuple[str, Path]] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _RUN_OUTPUT_DIRS and not d.startswith(".")]
        here = Path(directory)
        for name in filenames:
            if name.startswith(".") or name.endswith(".pyc"):
                continue
            if here == root and name in _RUN_OUTPUT_FILES:
                continue
            path = here / name
            found.append((path.relative_to(root).as_posix(), path))
    return iter(sorted(found))
