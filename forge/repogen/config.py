"""Limits for the repo pipeline, read from env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from forge.repogen.github_mcp import RepoConfigError

_DEFAULT_MAX_TARBALL_MB = 200
_DEFAULT_MIN_STABLE_TESTS = 10


@dataclass(frozen=True)
class RepoLimits:
    max_tarball_bytes: int
    min_stable_tests: int


def repo_limits_from_env() -> RepoLimits:
    return RepoLimits(
        max_tarball_bytes=_int("FORGE_REPO_MAX_TARBALL_MB", _DEFAULT_MAX_TARBALL_MB, minimum=1) * 1_000_000,
        min_stable_tests=_int("FORGE_REPO_MIN_STABLE_TESTS", _DEFAULT_MIN_STABLE_TESTS, minimum=0),
    )


def _int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name) or str(default)
    try:
        value = int(raw)
    except ValueError:
        value = minimum - 1
    if value < minimum:
        raise RepoConfigError(f"{name} must be an integer of at least {minimum}, got {raw!r}")
    return value
