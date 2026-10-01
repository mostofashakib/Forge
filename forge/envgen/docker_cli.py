"""The one place Forge shells out to the `docker` CLI.

The CLI is used instead of the Python SDK for builds, commits and pulls: the
SDK's credential-helper resolution fails when an optional helper (for example
docker-credential-gcloud) is configured but not authenticated.
"""
from __future__ import annotations

import subprocess
from typing import Protocol

# Generous enough for a cold pip install, but a hung build must not hold a
# worker slot forever.
BUILD_TIMEOUT_S = 900


class DockerCLI(Protocol):
    """Runs docker commands. Swap in a fake to test without a daemon."""

    def run(self, *args: str, timeout: float = BUILD_TIMEOUT_S) -> None:
        """Run `docker <args>`, raising RuntimeError with Docker's output on failure."""
        ...

    def succeeds(self, *args: str) -> bool:
        """Whether `docker <args>` exits 0. Never raises."""
        ...

    def discard(self, *args: str) -> None:
        """Best-effort cleanup, such as `rm -f` or `rmi -f`. Failures are ignored."""
        ...


class SubprocessDockerCLI:
    """`DockerCLI` backed by the local docker binary."""

    def run(self, *args: str, timeout: float = BUILD_TIMEOUT_S) -> None:
        try:
            subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"docker {args[0]} timed out after {timeout}s") from exc
        except subprocess.CalledProcessError as exc:
            output = (exc.stderr or exc.stdout or "(no output)").strip()
            raise RuntimeError(f"docker {args[0]} failed (exit {exc.returncode}):\n{output}") from exc

    def succeeds(self, *args: str) -> bool:
        try:
            result = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
        except Exception:
            return False
        return result.returncode == 0

    def discard(self, *args: str) -> None:
        subprocess.run(["docker", *args], capture_output=True, check=False)
