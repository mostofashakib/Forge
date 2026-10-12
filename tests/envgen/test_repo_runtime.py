"""Repo environment images run as shells, like the CLI image, with no port."""
from unittest.mock import MagicMock, patch

import docker.errors
import pytest

from forge.envgen.container import ContainerRuntime
from forge.envgen.docker_images import REPO_IMAGE_PREFIX, is_shell_image
from tests.envgen.fake_docker import FakeDocker

REPO_IMAGE = f"{REPO_IMAGE_PREFIX}acme-widgets:latest"


@pytest.mark.parametrize("tag,expected", [
    ("builtin:cli", True),
    (REPO_IMAGE, True),
    ("builtin:browser", False),
    ("forge-env-app:latest", False),
    (None, False),
])
def test_shell_images_are_the_cli_and_repo_images(tag, expected):
    assert is_shell_image(tag) is expected


def test_run_repo_launches_the_repo_image_with_cli_isolation():
    daemon = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=daemon):
        container_id, port = ContainerRuntime().run_repo("acme_widgets", REPO_IMAGE)
    (container,) = daemon.containers._by_id.values()
    assert port == 0 and container.id == container_id
    assert container.kwargs["image"] == REPO_IMAGE
    assert container.kwargs["network_mode"] == "none"
    assert container.kwargs["command"] == ["tail", "-f", "/dev/null"]
    assert container.kwargs["environment"]["LD_PRELOAD"].endswith("libfaketime.so.1")


def test_start_runs_a_removed_repo_env_fresh_without_needing_a_port():
    daemon = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=daemon):
        container_id, port = ContainerRuntime().start("acme_widgets", "", REPO_IMAGE)
    assert port == 0 and container_id


def test_start_restarts_a_stopped_repo_env_and_reports_no_port():
    existing = MagicMock(id="repo-abc")
    client = MagicMock()
    client.containers.get.return_value = existing
    with patch("forge.envgen.container.docker.from_env", return_value=client):
        assert ContainerRuntime().start("acme_widgets", "repo-abc", REPO_IMAGE) == ("repo-abc", 0)
    existing.start.assert_called_once()


def test_start_with_a_pruned_repo_image_explains_the_rebuild():
    client = MagicMock()
    client.containers.get.side_effect = docker.errors.NotFound("gone")
    client.containers.run.side_effect = docker.errors.ImageNotFound("pruned")
    with patch("forge.envgen.container.docker.from_env", return_value=client):
        with pytest.raises(RuntimeError, match="must be rebuilt"):
            ContainerRuntime().start("acme_widgets", "", REPO_IMAGE)
