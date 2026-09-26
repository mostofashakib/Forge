"""CLI grading runs outside the agent's writable space.

The agent is root in its shell, so anything inside that container, including
`grep`, `bash`, and `/etc/ld.so.preload`, may have been rewritten. Grading
snapshots the finished container, starts a fresh copy with no network and no
agent processes, and runs each assertion through a static BusyBox toolbox
mounted read-only from a pinned image the agent never touched.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from forge.envgen.container import (
    FORGE_GRADER_TOOLBOX_IMAGE,
    ensure_grader_toolbox,
    grading_sandbox,
)


class _DockerCli:
    """Records docker CLI calls. `fail` makes a matching subcommand fail."""

    def __init__(self, *, volume_exists: bool = True, fail: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self._volume_exists = volume_exists
        self._fail = fail

    def __call__(self, cmd, **_kwargs):
        self.calls.append(cmd)
        if self._fail and cmd[1] == self._fail:
            raise subprocess.CalledProcessError(1, cmd, stderr=f"{self._fail} failed")
        if cmd[1:3] == ["volume", "inspect"]:
            return MagicMock(returncode=0 if self._volume_exists else 1, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    def sub(self, name: str) -> list[list[str]]:
        return [c for c in self.calls if c[1] == name]


def _grade(cli: _DockerCli, raise_inside: bool = False) -> list[str]:
    with patch("forge.envgen.container.subprocess.run", side_effect=cli), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        with grading_sandbox("agentcontainer123456") as exec_argv:
            if raise_inside:
                raise RuntimeError("assertion runner crashed")
            return exec_argv


def test_grading_runs_in_a_fresh_snapshot_with_no_network():
    cli = _DockerCli()
    _grade(cli)

    [commit] = cli.sub("commit")
    assert commit[2] == "agentcontainer123456"
    [run] = cli.sub("run")
    assert run[run.index("--network") + 1] == "none"
    assert run[-2] == commit[3]  # the grader starts from the snapshot image


def test_assertions_use_the_read_only_trusted_toolbox_with_a_clean_environment():
    cli = _DockerCli()
    exec_argv = _grade(cli)

    [run] = cli.sub("run")
    mount = run[run.index("-v") + 1]
    assert mount.endswith(":/opt/forge-grader:ro")
    assert run[run.index("--entrypoint") + 1] == "/opt/forge-grader/bin/sleep"
    assert exec_argv[:2] == ["docker", "exec"]
    assert exec_argv[3:5] == ["/opt/forge-grader/bin/env", "-i"]
    assert exec_argv[-2:] == ["/opt/forge-grader/bin/sh", "-c"]
    path = next(arg for arg in exec_argv if arg.startswith("PATH="))
    assert path.startswith("PATH=/opt/forge-grader/bin:")


def test_assertions_never_execute_in_the_agents_own_container():
    cli = _DockerCli()
    exec_argv = _grade(cli)

    assert exec_argv[2] != "agentcontainer123456"


def test_the_grader_and_snapshot_are_removed_even_when_grading_fails():
    cli = _DockerCli()
    with pytest.raises(RuntimeError, match="assertion runner crashed"):
        _grade(cli, raise_inside=True)

    assert cli.sub("rm") and cli.sub("rmi")


def test_the_toolbox_volume_is_populated_once_from_the_pinned_image():
    cli = _DockerCli(volume_exists=False)
    with patch("forge.envgen.container.subprocess.run", side_effect=cli), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        volume = ensure_grader_toolbox()

    assert cli.sub("volume")[1][2] == "create"
    [populate] = cli.sub("run")
    assert FORGE_GRADER_TOOLBOX_IMAGE in populate
    assert f"{volume}:/opt/forge-grader" in populate


def test_an_existing_toolbox_volume_is_not_rebuilt():
    cli = _DockerCli(volume_exists=True)
    with patch("forge.envgen.container.subprocess.run", side_effect=cli):
        ensure_grader_toolbox()

    assert cli.sub("run") == []


def test_a_failed_toolbox_population_leaves_no_half_built_volume():
    # The volume's existence is what marks the toolbox ready, so a failed
    # copy must not leave an empty volume behind.
    cli = _DockerCli(volume_exists=False, fail="run")
    with patch("forge.envgen.container.subprocess.run", side_effect=cli), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        with pytest.raises(RuntimeError, match="grader toolbox"):
            ensure_grader_toolbox()

    assert ["volume", "rm"] in [c[1:3] for c in cli.calls]
