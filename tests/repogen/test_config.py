"""Repo pipeline limits come from env and fail clearly when malformed."""
import pytest

from forge.repogen.config import RepoLimits, repo_limits_from_env
from forge.repogen.github_mcp import RepoConfigError


def test_defaults(monkeypatch):
    monkeypatch.delenv("FORGE_REPO_MAX_TARBALL_MB", raising=False)
    monkeypatch.delenv("FORGE_REPO_MIN_STABLE_TESTS", raising=False)
    assert repo_limits_from_env() == RepoLimits(max_tarball_bytes=200_000_000, min_stable_tests=10)


def test_overrides(monkeypatch):
    monkeypatch.setenv("FORGE_REPO_MAX_TARBALL_MB", "50")
    monkeypatch.setenv("FORGE_REPO_MIN_STABLE_TESTS", "3")
    assert repo_limits_from_env() == RepoLimits(max_tarball_bytes=50_000_000, min_stable_tests=3)


@pytest.mark.parametrize("name,value", [
    ("FORGE_REPO_MAX_TARBALL_MB", "big"),
    ("FORGE_REPO_MAX_TARBALL_MB", "0"),
    ("FORGE_REPO_MIN_STABLE_TESTS", "-1"),
])
def test_malformed_values_name_the_variable(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RepoConfigError, match=name):
        repo_limits_from_env()
