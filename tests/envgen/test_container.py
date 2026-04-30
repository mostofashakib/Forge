import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_build_writes_dockerfile_and_calls_docker_build(tmp_path):
    from forge.envgen.container import ContainerRuntime
    with patch("forge.envgen.container.docker") as mock_docker:
        mock_client = MagicMock()
        mock_image = MagicMock()
        mock_image.tags = ["forge-env:test_env:latest"]
        mock_client.images.build.return_value = (mock_image, iter([]))
        mock_docker.from_env.return_value = mock_client

        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "main.py").write_text("# app")

        runtime = ContainerRuntime()
        tag = runtime.build("test_env", app_dir)

        assert (app_dir / "Dockerfile").exists()
        mock_client.images.build.assert_called_once()
        assert tag == "forge-env:test_env:latest"


def test_run_returns_container_id_and_port():
    from forge.envgen.container import ContainerRuntime
    with patch("forge.envgen.container.docker") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_container.ports = {"8000/tcp": [{"HostPort": "54321"}]}
        mock_client.containers.run.return_value = mock_container
        mock_docker.from_env.return_value = mock_client

        runtime = ContainerRuntime()
        container_id, port = runtime.run("test_env", "forge-env:test_env:latest")

        assert container_id == "abc123"
        assert port == 54321
        mock_client.containers.run.assert_called_once()


def test_stop_ignores_not_found():
    from forge.envgen.container import ContainerRuntime
    import docker.errors
    with patch("forge.envgen.container.docker") as mock_docker:
        mock_client = MagicMock()
        mock_client.containers.get.side_effect = docker.errors.NotFound("not found")
        mock_docker.from_env.return_value = mock_docker
        mock_docker.errors = docker.errors

        runtime = ContainerRuntime()
        runtime._docker = mock_client
        runtime.stop("nonexistent")  # should not raise


def test_reattach_all_returns_managed_containers():
    from forge.envgen.container import ContainerRuntime
    with patch("forge.envgen.container.docker") as mock_docker:
        mock_client = MagicMock()
        mock_c = MagicMock()
        mock_c.id = "xyz"
        mock_c.labels = {"forge.env": "my_env"}
        mock_c.ports = {"8000/tcp": [{"HostPort": "9999"}]}
        mock_client.containers.list.return_value = [mock_c]
        mock_docker.from_env.return_value = mock_client

        runtime = ContainerRuntime()
        result = runtime.reattach_all()

        assert result == [("my_env", "xyz", 9999)]
