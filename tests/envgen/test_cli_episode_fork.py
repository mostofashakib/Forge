"""Every CLI episode starts from the same snapshot, and starts warm.

A CLI environment used to run every episode in its one long-lived shell, so
episode two inherited episode one's files. Now a run snapshots the
environment once, each episode forks a fresh container from that snapshot,
and a pre-started spare means the next episode never waits for a boot.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from forge.envgen.container import ContainerRuntime, cli_snapshot_tag
from tests.envgen.fake_docker import FakeDocker

SNAPSHOT = cli_snapshot_tag("shell", "run-1")


@pytest.fixture
def daemon():
    fake = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=fake), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        yield fake


def _containers(daemon: FakeDocker) -> dict[str, object]:
    return {c.name: c for c in daemon.containers._by_id.values()}


def test_a_run_snapshots_the_environment_once_under_a_run_scoped_tag():
    with patch("forge.envgen.container._docker_cli") as cli:
        tag = ContainerRuntime().snapshot_cli("shell", "base-container", "run-1")

    assert tag == SNAPSHOT
    cli.assert_called_once_with("commit", "base-container", SNAPSHOT)


def test_a_new_run_gets_a_new_snapshot_tag():
    assert cli_snapshot_tag("shell", "run-1") != cli_snapshot_tag("shell", "run-2")


def test_an_episode_takes_the_warm_container_without_booting_one(daemon):
    runtime = ContainerRuntime()
    warm_id = runtime.warm_cli("shell", SNAPSHOT)

    runs_before = next(daemon.id_counter)
    with runtime.cli_episode("shell", SNAPSHOT, refill=False) as episode_id:
        runs_during = next(daemon.id_counter)
        assert episode_id == warm_id

    assert runs_during == runs_before + 1  # no container was created on entry


def test_an_episode_forks_from_the_snapshot_when_nothing_is_warm(daemon):
    with ContainerRuntime().cli_episode("shell", SNAPSHOT, refill=False) as episode_id:
        episode = daemon.containers.get(episode_id)
        assert episode.kwargs["image"] == SNAPSHOT


def test_forked_episodes_keep_the_cli_isolation(daemon):
    with ContainerRuntime().cli_episode("shell", SNAPSHOT, refill=False) as episode_id:
        kwargs = daemon.containers.get(episode_id).kwargs

    assert kwargs["network_mode"] == "none"
    assert kwargs["environment"]["FAKETIME"] == "@2023-11-14 22:13:20"
    assert kwargs["labels"]["forge.env"] == "shell"


def test_a_warm_container_from_an_older_snapshot_is_never_used(daemon):
    runtime = ContainerRuntime()
    stale_id = runtime.warm_cli("shell", cli_snapshot_tag("shell", "run-0"))

    with runtime.cli_episode("shell", SNAPSHOT, refill=False) as episode_id:
        assert episode_id != stale_id
        assert daemon.containers.get(episode_id).kwargs["image"] == SNAPSHOT


def test_each_episode_container_is_discarded_so_nothing_leaks_forward(daemon):
    runtime = ContainerRuntime()
    with runtime.cli_episode("shell", SNAPSHOT, refill=True) as first:
        pass
    with runtime.cli_episode("shell", SNAPSHOT, refill=False) as second:
        pass

    assert first != second
    assert first not in daemon.containers._by_id and second not in daemon.containers._by_id


def test_refill_leaves_a_warm_container_for_the_next_episode(daemon):
    runtime = ContainerRuntime()
    with runtime.cli_episode("shell", SNAPSHOT, refill=True):
        pass

    warm = _containers(daemon)["forge-shell-warm"]
    assert warm.status == "running"
    assert warm.kwargs["image"] == SNAPSHOT


def test_no_refill_after_the_last_episode_leaves_nothing_behind(daemon):
    with ContainerRuntime().cli_episode("shell", SNAPSHOT, refill=False):
        pass

    assert "forge-shell-warm" not in _containers(daemon)


def test_removing_the_environment_also_removes_its_warm_and_episode_containers(daemon):
    runtime = ContainerRuntime()
    runtime.warm_cli("shell", SNAPSHOT)

    runtime._remove_existing("shell")

    assert not any(name.startswith("forge-shell") for name in _containers(daemon))


def test_discarding_a_snapshot_removes_its_image_and_tolerates_a_missing_one():
    with patch("forge.envgen.container.subprocess.run") as run:
        ContainerRuntime.discard_cli_snapshot(SNAPSHOT)

    assert run.call_args.args[0] == ["docker", "rmi", "-f", SNAPSHOT]
    assert run.call_args.kwargs["check"] is False
