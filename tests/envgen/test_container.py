import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import docker.errors

from forge.envgen.container import ContainerRuntime


# ---------------------------------------------------------------------------
# _container_name — sanitization
# ---------------------------------------------------------------------------

def test_container_name_alphanumeric():
    assert ContainerRuntime._container_name("myenv") == "forge-myenv"


def test_container_name_underscores_preserved():
    assert ContainerRuntime._container_name("my_env") == "forge-my_env"


def test_container_name_hyphens_preserved():
    assert ContainerRuntime._container_name("my-env") == "forge-my-env"


def test_container_name_spaces_replaced():
    assert ContainerRuntime._container_name("my env") == "forge-my-env"


def test_container_name_special_chars_replaced():
    assert ContainerRuntime._container_name("env@v1!") == "forge-env-v1-"


# ---------------------------------------------------------------------------
# build — uses subprocess CLI, not Docker SDK
# ---------------------------------------------------------------------------

def test_build_writes_dockerfile_and_calls_subprocess(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text("# app")

    with patch("forge.envgen.container.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        runtime = ContainerRuntime()
        tag = runtime.build("test_env", app_dir)

    assert (app_dir / "Dockerfile").exists()
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[0] == "docker"
    assert args[1] == "build"
    assert "-t" in args
    assert tag == "forge-env-test-env:latest"


def test_build_does_not_use_docker_sdk(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text("# app")

    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container.docker.from_env") as mock_sdk:
        mock_run.return_value = MagicMock(returncode=0)
        ContainerRuntime().build("test_env", app_dir)

    mock_sdk.assert_not_called()


# ---------------------------------------------------------------------------
# run_cli — subprocess pull + SDK container create
# ---------------------------------------------------------------------------

def test_run_cli_pulls_image_via_subprocess():
    mock_container = MagicMock()
    mock_container.id = "cli-abc"
    mock_docker = MagicMock()
    mock_docker.containers.get.side_effect = docker.errors.NotFound("not found")
    mock_docker.containers.run.return_value = mock_container

    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        mock_run.return_value = MagicMock(returncode=0)
        runtime = ContainerRuntime()
        container_id, port = runtime.run_cli("my_env")

    mock_run.assert_called_once_with(
        ["docker", "pull", "ubuntu:22.04"],
        check=True, capture_output=True, text=True,
    )
    assert container_id == "cli-abc"
    assert port == 0


def test_run_cli_container_uses_tail_command():
    mock_container = MagicMock()
    mock_container.id = "cli-xyz"
    mock_docker = MagicMock()
    mock_docker.containers.get.side_effect = docker.errors.NotFound("not found")
    mock_docker.containers.run.return_value = mock_container

    with patch("forge.envgen.container.subprocess.run"), \
         patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        ContainerRuntime().run_cli("keepalive_env")

    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["image"] == "ubuntu:22.04"
    assert kwargs["command"] == ["tail", "-f", "/dev/null"]
    assert kwargs["detach"] is True
    assert kwargs["labels"]["forge.type"] == "cli"


# ---------------------------------------------------------------------------
# run_browser — subprocess pull + SDK container create
# ---------------------------------------------------------------------------

def test_run_browser_pulls_image_via_subprocess():
    mock_container = MagicMock()
    mock_container.id = "browser-abc"
    mock_container.ports = {"3001/tcp": [{"HostPort": "45678"}]}
    mock_docker = MagicMock()
    mock_docker.containers.get.side_effect = docker.errors.NotFound("not found")
    mock_docker.containers.run.return_value = mock_container

    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        mock_run.return_value = MagicMock(returncode=0)
        container_id, port = ContainerRuntime().run_browser("my_browser_env")

    mock_run.assert_called_once_with(
        ["docker", "pull", "lscr.io/linuxserver/chromium:latest"],
        check=True, capture_output=True, text=True,
    )
    assert container_id == "browser-abc"
    assert port == 45678


# ---------------------------------------------------------------------------
# run (general) — uses Docker SDK directly (no subprocess pull)
# ---------------------------------------------------------------------------

def test_run_returns_container_id_and_port():
    mock_container = MagicMock()
    mock_container.id = "abc123"
    mock_container.ports = {"8000/tcp": [{"HostPort": "54321"}]}
    mock_docker = MagicMock()
    mock_docker.containers.run.return_value = mock_container

    with patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        container_id, port = ContainerRuntime().run("test_env", "forge-env-test-env:latest")

    assert container_id == "abc123"
    assert port == 54321
    mock_docker.containers.run.assert_called_once()


# ---------------------------------------------------------------------------
# stop — ignores NotFound
# ---------------------------------------------------------------------------

def test_stop_ignores_not_found():
    mock_docker = MagicMock()
    mock_docker.containers.get.side_effect = docker.errors.NotFound("not found")

    with patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        ContainerRuntime().stop("nonexistent")  # must not raise


def test_stop_calls_stop_on_container():
    mock_container = MagicMock()
    mock_docker = MagicMock()
    mock_docker.containers.get.return_value = mock_container

    with patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        ContainerRuntime().stop("running_container_id")

    mock_container.stop.assert_called_once_with(timeout=10)


# ---------------------------------------------------------------------------
# reattach_all — lists managed containers
# ---------------------------------------------------------------------------

def test_reattach_all_returns_managed_containers():
    mock_c = MagicMock()
    mock_c.id = "xyz"
    mock_c.labels = {"forge.env": "my_env"}
    mock_c.ports = {"8000/tcp": [{"HostPort": "9999"}]}
    mock_docker = MagicMock()
    mock_docker.containers.list.return_value = [mock_c]

    with patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        result = ContainerRuntime().reattach_all()

    assert result == [("my_env", "xyz", 9999)]


def test_reattach_all_skips_containers_without_port():
    mock_c = MagicMock()
    mock_c.id = "no-port"
    mock_c.labels = {"forge.env": "portless_env"}
    mock_c.ports = {}
    mock_docker = MagicMock()
    mock_docker.containers.list.return_value = [mock_c]

    with patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        result = ContainerRuntime().reattach_all()

    assert result == []
