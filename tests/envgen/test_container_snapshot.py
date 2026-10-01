"""Unit tests for initial state snapshots and cloned episode containers (Feature Request 5)."""
from __future__ import annotations

from unittest.mock import patch
import pytest

from forge.envgen.container import (
    ContainerRuntime,
    initial_snapshot_tag,
)
from tests.envgen.fake_docker import FakeDocker


@pytest.fixture
def daemon():
    fake = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=fake), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        yield fake


def _containers(daemon: FakeDocker) -> dict[str, object]:
    return {c.name: c for c in daemon.containers._by_id.values()}


def test_initial_snapshot_tag_deterministic():
    tag1 = initial_snapshot_tag("my_app", seed=42)
    tag2 = initial_snapshot_tag("my_app", seed=42)
    tag_diff_seed = initial_snapshot_tag("my_app", seed=99)
    assert tag1 == "forge-snapshot-my-app:seed-42"
    assert tag1 == tag2
    assert tag1 != tag_diff_seed


def test_snapshot_initial_state_commits_container():
    with patch("forge.envgen.container._docker_cli") as cli:
        tag = ContainerRuntime().snapshot_initial_state("my_app", "base-cont-123", seed=42)
    assert tag == "forge-snapshot-my-app:seed-42"
    cli.assert_called_once_with("commit", "base-cont-123", tag)


def test_discard_initial_snapshot():
    with patch("subprocess.run") as run_mock:
        ContainerRuntime.discard_initial_snapshot("forge-snapshot-my-app:seed-42")
    run_mock.assert_called_once()
    assert "rmi" in run_mock.call_args[0][0]


def test_cloned_episode_general_http(daemon):
    runtime = ContainerRuntime()
    snapshot = initial_snapshot_tag("webapp", seed=0)

    with runtime.cloned_episode("webapp", snapshot, env_type="general", episode_id="ep1") as (cid, port):
        assert cid in daemon.containers._by_id
        ep_container = daemon.containers.get(cid)
        assert ep_container.kwargs["image"] == snapshot
        assert ep_container.labels["forge.role"] == "episode"
        assert port > 0
        # Gateway exists while episode is active
        assert "forge-webapp-ep-ep1-gw" in _containers(daemon)

    # After episode completion, cloned container and ephemeral gateway are destroyed
    assert cid not in daemon.containers._by_id
    assert "forge-webapp-ep-ep1-gw" not in _containers(daemon)


def test_cloned_episode_browser(daemon):
    runtime = ContainerRuntime()
    snapshot = initial_snapshot_tag("browserapp", seed=7)

    with runtime.cloned_episode("browserapp", snapshot, env_type="browser", episode_id="ep_web") as (cid, port):
        assert cid in daemon.containers._by_id
        ep_container = daemon.containers.get(cid)
        assert ep_container.kwargs["image"] == snapshot
        assert ep_container.labels["forge.role"] == "episode"
        assert port > 0
        assert "forge-browserapp-ep-ep_web-gw" in _containers(daemon)

    # Cleaned up
    assert cid not in daemon.containers._by_id
    assert "forge-browserapp-ep-ep_web-gw" not in _containers(daemon)


def test_cloned_episode_cli_delegates(daemon):
    runtime = ContainerRuntime()
    snapshot = initial_snapshot_tag("shell", seed=1)

    with runtime.cloned_episode("shell", snapshot, env_type="cli", refill=False) as cid:
        assert cid in daemon.containers._by_id
        ep_container = daemon.containers.get(cid)
        assert ep_container.kwargs["image"] == snapshot

    assert cid not in daemon.containers._by_id
