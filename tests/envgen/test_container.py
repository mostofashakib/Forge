import logging
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import docker.errors
import pytest

from tests.envgen.fake_docker import FakeDocker

from forge.envgen.container import (
    ContainerRuntime,
    FORGE_APP_PORT,
    FORGE_PYTHON_BASE,
    STANDARD_BASE_IMAGES,
    CLI_DOCKERFILE,
    CLI_RUNTIME_IMAGE,
    _cli_image_tag,
    ensure_cli_image,
    FORGE_BROWSER_IMAGE,
    FORGE_CLI_IMAGE,
    RUNTIME_LOCK,
    _HUB_MIRRORS,
    _image_cached_locally,
    _is_hub_image,
    _mirror_ref_for,
    _normalise_dockerfile_base,
    _normalise_dockerfile_install,
    _normalise_dockerfile_port,
    _parse_from_image,
    _pull_with_retry,
    _wait_for_port_binding,
    _write_locked_requirements,
    prewarm_standard_base_images,
    pull_image,
)


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
# _parse_from_image — Dockerfile FROM parsing
# ---------------------------------------------------------------------------

def test_parse_from_standard(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    assert _parse_from_image(df) == "python:3.12-slim"


def test_parse_from_llm_version(tmp_path):
    """LLM often generates python:3.11-slim — must be parsed correctly."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.11-slim\nWORKDIR /app\n")
    assert _parse_from_image(df) == "python:3.11-slim"


def test_parse_from_strips_build_stage_alias(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12-slim AS builder\nWORKDIR /app\n")
    assert _parse_from_image(df) == "python:3.12-slim"


def test_parse_from_scratch_returns_none(tmp_path):
    """scratch has no registry — no pull needed."""
    df = tmp_path / "Dockerfile"
    df.write_text("FROM scratch\nCOPY binary /\n")
    assert _parse_from_image(df) is None


def test_parse_from_ubuntu(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM ubuntu:22.04\nRUN apt-get update\n")
    assert _parse_from_image(df) == "ubuntu:22.04"


def test_parse_from_takes_first_from_in_multistage(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim AS base\n"
        "FROM node:20-slim AS frontend\n"
        "FROM base AS final\n"
    )
    assert _parse_from_image(df) == "python:3.12-slim"


def test_parse_from_missing_returns_none(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text("# no FROM here\nRUN echo hello\n")
    assert _parse_from_image(df) is None


# ---------------------------------------------------------------------------
# _pull_with_retry — retry logic for transient network errors
# ---------------------------------------------------------------------------

def test_pull_with_retry_succeeds_on_first_attempt():
    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container.time.sleep"):
        mock_run.return_value = MagicMock(returncode=0)
        _pull_with_retry("python:3.12-slim")

    mock_run.assert_called_once_with(
        ["docker", "pull", "python:3.12-slim"],
        check=True, capture_output=True, text=True, timeout=120,
    )


def test_pull_with_retry_timeout_is_passed_to_subprocess():
    """Custom pull_timeout is forwarded to subprocess.run."""
    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container.time.sleep"):
        mock_run.return_value = MagicMock(returncode=0)
        _pull_with_retry("python:3.12-slim", pull_timeout=60)

    _, kwargs = mock_run.call_args
    assert kwargs["timeout"] == 60


def test_pull_with_retry_retries_on_timeout_expired():
    """TimeoutExpired is a transient error — must retry, not fail immediately."""
    timeout_exc = subprocess.TimeoutExpired(["docker", "pull"], 120)
    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container.time.sleep") as mock_sleep:
        mock_run.side_effect = [timeout_exc, MagicMock(returncode=0)]
        _pull_with_retry("python:3.12-slim", max_attempts=3)

    assert mock_run.call_count == 2
    mock_sleep.assert_called_once_with(1)


def test_pull_with_retry_timeout_message_in_error():
    """When all attempts time out the RuntimeError must mention the timeout."""
    timeout_exc = subprocess.TimeoutExpired(["docker", "pull"], 120)
    with patch("forge.envgen.container.subprocess.run", side_effect=timeout_exc), \
         patch("forge.envgen.container.time.sleep"):
        with pytest.raises(RuntimeError, match="timed out"):
            _pull_with_retry("python:3.12-slim", max_attempts=2)


def test_pull_with_retry_retries_on_eof_then_succeeds():
    """Simulates an EOF on the first pull attempt, success on the second."""
    eof_exc = subprocess.CalledProcessError(1, "docker pull")
    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container.time.sleep") as mock_sleep:
        mock_run.side_effect = [eof_exc, MagicMock(returncode=0)]
        _pull_with_retry("python:3.12-slim", max_attempts=3)

    assert mock_run.call_count == 2
    mock_sleep.assert_called_once_with(1)  # 2^0 = 1 s backoff after attempt 0


def test_pull_with_retry_raises_after_all_attempts_exhausted():
    """All attempts fail → RuntimeError is raised, not CalledProcessError."""
    eof_exc = subprocess.CalledProcessError(1, "docker pull")
    with patch("forge.envgen.container.subprocess.run", side_effect=eof_exc), \
         patch("forge.envgen.container.time.sleep"):
        with pytest.raises(RuntimeError, match="Failed to pull python:3.12-slim after 3 attempts"):
            _pull_with_retry("python:3.12-slim", max_attempts=3)


def test_pull_with_retry_exponential_backoff():
    """Backoff delays must be 1 s, 2 s (2^0, 2^1) for a 3-attempt run."""
    eof_exc = subprocess.CalledProcessError(1, "docker pull")
    delays = []
    with patch("forge.envgen.container.subprocess.run", side_effect=eof_exc), \
         patch("forge.envgen.container.time.sleep", side_effect=lambda d: delays.append(d)):
        with pytest.raises(RuntimeError):
            _pull_with_retry("python:3.12-slim", max_attempts=3)

    assert delays == [1, 2]  # last attempt has no sleep


def test_pull_with_retry_surfaces_docker_stderr_in_error():
    """Docker's stderr must appear in the RuntimeError message for diagnosis."""
    exc_with_output = subprocess.CalledProcessError(
        1, "docker pull", stderr="Error response from daemon: EOF"
    )
    with patch("forge.envgen.container.subprocess.run", side_effect=exc_with_output), \
         patch("forge.envgen.container.time.sleep"):
        with pytest.raises(RuntimeError, match="Error response from daemon: EOF"):
            _pull_with_retry("python:3.12-slim", max_attempts=1)


def test_pull_with_retry_fails_fast_on_image_not_found():
    """'not found' is a permanent error — must not retry, must still raise RuntimeError."""
    exc_not_found = subprocess.CalledProcessError(
        1, "docker pull",
        stderr="Error response from daemon: manifest for python:0.0.0 not found",
    )
    with patch("forge.envgen.container.subprocess.run", side_effect=exc_not_found) as mock_run, \
         patch("forge.envgen.container.time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="not found"):
            _pull_with_retry("python:0.0.0", max_attempts=3)

    # Only one attempt — no retries for permanent errors
    assert mock_run.call_count == 1
    mock_sleep.assert_not_called()


def test_pull_with_retry_fails_fast_on_auth_denied():
    """'pull access denied' is a permanent error — must not retry."""
    exc_denied = subprocess.CalledProcessError(
        1, "docker pull",
        stderr="Error response from daemon: pull access denied for private/image",
    )
    with patch("forge.envgen.container.subprocess.run", side_effect=exc_denied) as mock_run, \
         patch("forge.envgen.container.time.sleep"):
        with pytest.raises(RuntimeError, match="Failed to pull"):
            _pull_with_retry("private/image", max_attempts=3)

    assert mock_run.call_count == 1


def test_pull_with_retry_eof_uses_capped_backoff():
    """EOF is transient — must retry up to max_attempts with backoff capped at 30 s."""
    eof_exc = subprocess.CalledProcessError(
        1, "docker pull",
        stderr="failed to do request: Head ...: EOF",
    )
    delays: list[float] = []
    with patch("forge.envgen.container.subprocess.run", side_effect=eof_exc), \
         patch("forge.envgen.container.time.sleep", side_effect=lambda d: delays.append(d)):
        with pytest.raises(RuntimeError, match="Failed to pull"):
            _pull_with_retry("python:3.11-slim", max_attempts=5)

    # 5 attempts → 4 sleeps; backoff: 1, 2, 4, 8 (all < 30 cap)
    assert delays == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# _image_cached_locally
# ---------------------------------------------------------------------------

def test_image_cached_locally_returns_true_when_inspect_succeeds():
    with patch("forge.envgen.container.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert _image_cached_locally("python:3.12-slim") is True

    mock_run.assert_called_once_with(
        ["docker", "image", "inspect", "python:3.12-slim"],
        capture_output=True, text=True, check=False,
    )


def test_image_cached_locally_returns_false_when_inspect_fails():
    with patch("forge.envgen.container.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        assert _image_cached_locally("python:3.11-slim") is False


def test_image_cached_locally_returns_false_on_exception():
    with patch("forge.envgen.container.subprocess.run", side_effect=OSError("docker not found")):
        assert _image_cached_locally("python:3.12-slim") is False


# ---------------------------------------------------------------------------
# build — uses subprocess CLI, not Docker SDK
# ---------------------------------------------------------------------------

def test_build_writes_dockerfile_and_calls_subprocess(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text("# app")

    # Image not cached → triggers a pull before docker build
    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container._image_cached_locally", return_value=False):
        mock_run.return_value = MagicMock(returncode=0)
        runtime = ContainerRuntime()
        tag = runtime.build("test_env", app_dir)

    assert (app_dir / "Dockerfile").exists()
    # Two subprocess calls: docker pull (base image) then docker build
    assert mock_run.call_count == 2
    pull_args = mock_run.call_args_list[0].args[0]
    assert pull_args[0] == "docker"
    assert pull_args[1] == "pull"
    build_args = mock_run.call_args_list[1].args[0]
    assert build_args[0] == "docker"
    assert build_args[1] == "build"
    assert "-t" in build_args
    assert tag == "forge-env-test-env:latest"


def test_build_skips_pull_when_image_cached_locally(tmp_path):
    """If the base image is in the local Docker cache, no pull should be attempted."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")

    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        mock_run.return_value = MagicMock(returncode=0)
        tag = ContainerRuntime().build("cached_env", app_dir)

    # Only docker build — no pull because image was already cached
    assert mock_run.call_count == 1
    assert mock_run.call_args.args[0][1] == "build"
    assert tag == "forge-env-cached-env:latest"


def test_build_prepulls_from_image_before_docker_build(tmp_path):
    """build() must pull the FROM image first (when not cached), then run docker build.

    The LLM-generated Dockerfile uses python:3.11-slim, but build() normalises
    that to FORGE_PYTHON_BASE before pulling — so the pull targets the
    canonical base, not the LLM's arbitrary choice.
    """
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\n"
    )

    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container._image_cached_locally", return_value=False):
        mock_run.return_value = MagicMock(returncode=0)
        ContainerRuntime().build("prepull_env", app_dir)

    assert mock_run.call_count == 2
    pull_call = mock_run.call_args_list[0]
    assert pull_call.args[0] == ["docker", "pull", FORGE_PYTHON_BASE]
    build_call = mock_run.call_args_list[1]
    assert build_call.args[0][1] == "build"


def test_build_skips_prepull_for_scratch(tmp_path):
    """FROM scratch has no registry image to pull."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM scratch\nCOPY binary /\n")

    with patch("forge.envgen.container.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        ContainerRuntime().build("scratch_env", app_dir)

    # Only the docker build call — no pull (scratch has no registry image)
    assert mock_run.call_count == 1
    assert mock_run.call_args.args[0][1] == "build"


def test_build_retries_pull_on_eof_then_builds(tmp_path):
    """EOF on first pull attempt is retried; docker build still runs on success."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")

    eof_exc = subprocess.CalledProcessError(1, "docker pull")

    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container.time.sleep"), \
         patch("forge.envgen.container._image_cached_locally", return_value=False):
        # pull fails once, then succeeds; build succeeds
        mock_run.side_effect = [eof_exc, MagicMock(returncode=0), MagicMock(returncode=0)]
        tag = ContainerRuntime().build("retry_env", app_dir)

    calls = [c.args[0] for c in mock_run.call_args_list]
    assert calls[0] == ["docker", "pull", FORGE_PYTHON_BASE]   # first pull (fails)
    assert calls[1] == ["docker", "pull", FORGE_PYTHON_BASE]   # retry (succeeds)
    assert calls[2][1] == "build"                                # docker build
    assert tag == "forge-env-retry-env:latest"


def test_build_does_not_run_docker_build_when_pull_fails_permanently(tmp_path):
    """If pull exhausts every transport (registries + direct HTTPS), docker
    build must NOT be attempted."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")

    eof_exc = subprocess.CalledProcessError(1, "docker pull")

    with patch("forge.envgen.container.subprocess.run", side_effect=eof_exc), \
         patch("forge.envgen.container.time.sleep"), \
         patch("forge.envgen._image_pull_http.pull_via_http",
               side_effect=RuntimeError("HTTPS also EOF")):
        with pytest.raises(RuntimeError, match="Failed to pull"):
            ContainerRuntime().build("nopull_env", app_dir)


def test_build_surfaces_docker_build_stderr_on_failure(tmp_path):
    """CalledProcessError from docker build is re-raised with stderr text."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM python:3.12-slim\nRUN false\n")

    build_exc = subprocess.CalledProcessError(
        1, "docker build", stderr="RUN false returned 1"
    )

    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container.time.sleep"):
        # pull succeeds, build fails
        mock_run.side_effect = [MagicMock(returncode=0), build_exc]
        with pytest.raises(RuntimeError, match="RUN false returned 1"):
            ContainerRuntime().build("failbuild_env", app_dir)


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

    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull, \
         patch("forge.envgen.container.subprocess.run", return_value=MagicMock(returncode=0)), \
         patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        runtime = ContainerRuntime()
        container_id, port = runtime.run_cli("my_env")

    mock_pull.assert_called_once_with(FORGE_CLI_IMAGE)
    assert container_id == "cli-abc"
    assert port == 0


def test_run_cli_skips_pull_when_image_cached():
    mock_container = MagicMock()
    mock_container.id = "cli-xyz"
    mock_docker = MagicMock()
    mock_docker.containers.get.side_effect = docker.errors.NotFound("not found")
    mock_docker.containers.run.return_value = mock_container

    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull, \
         patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        ContainerRuntime().run_cli("my_env")

    mock_pull.assert_not_called()


def test_run_cli_container_uses_tail_command():
    mock_container = MagicMock()
    mock_container.id = "cli-xyz"
    mock_docker = MagicMock()
    mock_docker.containers.get.side_effect = docker.errors.NotFound("not found")
    mock_docker.containers.run.return_value = mock_container

    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        ContainerRuntime().run_cli("keepalive_env")

    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["image"] == CLI_RUNTIME_IMAGE
    assert kwargs["command"] == ["tail", "-f", "/dev/null"]
    assert kwargs["detach"] is True
    assert kwargs["labels"]["forge.type"] == "cli"
    assert kwargs["pids_limit"] == 256
    assert kwargs["init"] is True


# ---------------------------------------------------------------------------
# run_browser — subprocess pull + SDK container create
# ---------------------------------------------------------------------------

def test_run_browser_pulls_image_via_subprocess():
    daemon = FakeDocker()
    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull, \
         patch("forge.envgen.container.docker.from_env", return_value=daemon):
        container_id, port = ContainerRuntime().run_browser("my_browser_env")

    pulled = [c.args[0] for c in mock_pull.call_args_list]
    assert FORGE_BROWSER_IMAGE in pulled
    assert daemon.containers.get(container_id).name == "forge-my_browser_env"
    assert port == int(daemon.named("forge-my_browser_env-gw").ports["3000/tcp"][0]["HostPort"])


def test_run_browser_skips_pull_when_image_cached():
    mock_container = MagicMock()
    mock_container.id = "browser-xyz"
    mock_container.ports = {"3000/tcp": [{"HostPort": "45679"}]}
    mock_docker = MagicMock()
    mock_docker.containers.get.side_effect = docker.errors.NotFound("not found")
    mock_docker.containers.run.return_value = mock_container

    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull, \
         patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        ContainerRuntime().run_browser("my_browser_env")

    mock_pull.assert_not_called()


# ---------------------------------------------------------------------------
# run (general) — uses Docker SDK directly (no subprocess pull)
# ---------------------------------------------------------------------------

def test_run_returns_container_id_and_port():
    daemon = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=daemon), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        container_id, port = ContainerRuntime().run("test_env", "forge-env-test-env:latest")

    app = daemon.containers.get(container_id)
    gateway = daemon.named("forge-test_env-gw")
    assert app.name == "forge-test_env"
    assert port == int(gateway.ports["8000/tcp"][0]["HostPort"])
    for container in (app, gateway):
        assert container.kwargs["cap_drop"] == ["ALL"]
        assert container.kwargs["security_opt"] == ["no-new-privileges:true"]


# ---------------------------------------------------------------------------
# stop — ignores NotFound
# ---------------------------------------------------------------------------

def test_stop_ignores_not_found():
    mock_docker = MagicMock()
    mock_docker.containers.get.side_effect = docker.errors.NotFound("not found")

    with patch("forge.envgen.container.docker.from_env", return_value=mock_docker):
        ContainerRuntime().stop("nonexistent")  # must not raise


def test_stop_calls_stop_on_container():
    daemon = FakeDocker()
    shell = daemon.containers.run(name="forge-shell", image="cli", labels={"forge.env": "shell", "forge.type": "cli"})

    with patch("forge.envgen.container.docker.from_env", return_value=daemon):
        ContainerRuntime().stop(shell.id)

    assert shell.status == "exited"


# ---------------------------------------------------------------------------
# reattach_all — lists managed containers
# ---------------------------------------------------------------------------

def test_reattach_all_returns_managed_containers():
    daemon = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=daemon), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        runtime = ContainerRuntime()
        app_id, port = runtime.run("my_env", "forge-env-my-env:latest")
        result = runtime.reattach_all()

    assert result == [("my_env", app_id, port)]


def test_reattach_all_uses_the_browser_ui_port():
    daemon = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=daemon), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        runtime = ContainerRuntime()
        browser_id, ui_port = runtime.run_browser("web_env")
        result = runtime.reattach_all()

    assert result == [("web_env", browser_id, ui_port)]


def test_reattach_all_skips_containers_without_port():
    # An app with no gateway, and a CLI shell, have no port to reattach.
    daemon = FakeDocker()
    daemon.containers.run(name="forge-portless_env", image="app",
                          labels={"forge.env": "portless_env", "forge.managed": "true"})
    daemon.containers.run(name="forge-shell", image="cli",
                          labels={"forge.env": "shell", "forge.managed": "true", "forge.type": "cli"})

    with patch("forge.envgen.container.docker.from_env", return_value=daemon):
        result = ContainerRuntime().reattach_all()

    assert result == []


# ---------------------------------------------------------------------------
# _normalise_dockerfile_base — rewrites LLM-chosen python tags to canonical base
# ---------------------------------------------------------------------------

def test_build_rejects_generated_network_import_before_docker(tmp_path: Path):
    (tmp_path / "main.py").write_text("import requests\n")
    (tmp_path / "Dockerfile").write_text(f"FROM {FORGE_PYTHON_BASE}\n")
    with pytest.raises(RuntimeError, match="violates network policy"):
        ContainerRuntime().build("unsafe_env", tmp_path)


def test_normalise_rewrites_python_311_to_canonical(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.11-slim\nWORKDIR /app\n")
    changed = _normalise_dockerfile_base(df)
    assert changed is True
    assert df.read_text() == f"FROM {FORGE_PYTHON_BASE}\nWORKDIR /app\n"


def test_normalise_rewrites_python_312_bookworm(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12-bookworm\n")
    changed = _normalise_dockerfile_base(df)
    assert changed is True
    assert df.read_text() == f"FROM {FORGE_PYTHON_BASE}\n"


def test_normalise_noop_when_already_canonical(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    df.write_text(f"FROM {FORGE_PYTHON_BASE}\nCOPY . .\n")
    changed = _normalise_dockerfile_base(df)
    assert changed is False
    assert df.read_text() == f"FROM {FORGE_PYTHON_BASE}\nCOPY . .\n"


def test_normalise_preserves_stage_alias(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.11-slim AS builder\nWORKDIR /app\n")
    changed = _normalise_dockerfile_base(df)
    assert changed is True
    assert df.read_text() == f"FROM {FORGE_PYTHON_BASE} AS builder\nWORKDIR /app\n"


def test_normalise_leaves_non_python_base_alone(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    df.write_text("FROM node:20-slim\nRUN npm install\n")
    changed = _normalise_dockerfile_base(df)
    assert changed is False
    assert df.read_text() == "FROM node:20-slim\nRUN npm install\n"


def test_normalise_handles_lowercase_from(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    df.write_text("from python:3.11-alpine\nWORKDIR /app\n")
    changed = _normalise_dockerfile_base(df)
    assert changed is True
    assert df.read_text() == f"from {FORGE_PYTHON_BASE}\nWORKDIR /app\n"


# ---------------------------------------------------------------------------
# _normalise_dockerfile_port — guardrail against the LLM picking a non-8000 port
# ---------------------------------------------------------------------------

def test_normalise_port_rewrites_expose_5000_to_canonical(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim\n"
        "EXPOSE 5000\n"
        "CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"5000\"]\n"
    )
    assert _normalise_dockerfile_port(df) is True
    text = df.read_text()
    assert f"EXPOSE {FORGE_APP_PORT}" in text
    assert "EXPOSE 5000" not in text
    assert f'"--port", "{FORGE_APP_PORT}"' in text
    assert '"5000"' not in text


def test_normalise_port_rewrites_shell_form_cmd(tmp_path: Path):
    """Shell-form `CMD uvicorn --port 8080` must also be rewritten."""
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim\n"
        "EXPOSE 8080\n"
        "CMD uvicorn main:app --host 0.0.0.0 --port 8080\n"
    )
    assert _normalise_dockerfile_port(df) is True
    text = df.read_text()
    assert f"--port {FORGE_APP_PORT}" in text
    assert "8080" not in text


def test_normalise_port_rewrites_equals_form(tmp_path: Path):
    """`--port=5000` form is also caught."""
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim\n"
        "EXPOSE 5000\n"
        "CMD uvicorn main:app --host=0.0.0.0 --port=5000\n"
    )
    assert _normalise_dockerfile_port(df) is True
    text = df.read_text()
    assert f"--port={FORGE_APP_PORT}" in text


def test_normalise_port_adds_expose_when_missing(tmp_path: Path):
    """If the LLM forgot EXPOSE entirely, we add EXPOSE 8000."""
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
    )
    assert _normalise_dockerfile_port(df) is True
    assert f"EXPOSE {FORGE_APP_PORT}" in df.read_text()


def test_normalise_port_noop_when_already_canonical(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    canonical = (
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        f"EXPOSE {FORGE_APP_PORT}\n"
        f'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{FORGE_APP_PORT}"]\n'
    )
    df.write_text(canonical)
    assert _normalise_dockerfile_port(df) is False
    assert df.read_text() == canonical


def test_normalise_port_handles_multiple_expose_lines(tmp_path: Path):
    """Some Dockerfiles publish more than one port (e.g. metrics on 9100).
    Every EXPOSE gets normalised — we only care about the app port being
    consistent. Extra published ports without our `-p` mapping are harmless."""
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim\n"
        "EXPOSE 5000\n"
        "EXPOSE 9100\n"
        "CMD uvicorn main:app --port 5000\n"
    )
    _normalise_dockerfile_port(df)
    text = df.read_text()
    # Both EXPOSE lines now point at the canonical port; CMD too.
    assert text.count(f"EXPOSE {FORGE_APP_PORT}") == 2
    assert "5000" not in text
    assert "9100" not in text


# ---------------------------------------------------------------------------
# Locked dependencies: every container installs the same hashed lock
# ---------------------------------------------------------------------------

_LOCKED_INSTALL = "RUN pip install --no-cache-dir --require-hashes -r requirements.txt"


def _requirement_blocks(lock_text: str) -> list[str]:
    """Each requirement with its continuation lines, comments dropped."""
    blocks: list[str] = []
    for line in lock_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith((" ", "\t")) and blocks:
            blocks[-1] += " " + stripped
        else:
            blocks.append(stripped)
    return blocks


def test_every_locked_requirement_is_pinned_and_hashed():
    blocks = _requirement_blocks(RUNTIME_LOCK.read_text())

    assert blocks, "the lock must not be empty"
    for block in blocks:
        name = block.split()[0]
        assert "==" in name, f"{name} is not pinned to one version"
        assert "--hash=sha256:" in block, f"{name} has no hash"


def test_the_lock_covers_every_package_the_apps_import():
    names = {block.split("==")[0].lower() for block in _requirement_blocks(RUNTIME_LOCK.read_text())}

    for package in ("fastapi", "uvicorn", "sqlalchemy", "redis", "httpx", "python-multipart", "pydantic"):
        assert package in names, f"{package} missing from the lock"


def test_build_replaces_llm_requirements_with_the_lock(tmp_path: Path):
    # Unpinned names resolve to whatever PyPI serves on build day, so the
    # LLM's list never reaches pip. The correctness gate rejects code that
    # imports anything the lock lacks.
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "requirements.txt").write_text("fastapi\nbeautifulsoup4==4.12.0\nrequests\n")

    _write_locked_requirements(app_dir)

    assert (app_dir / "requirements.txt").read_text() == RUNTIME_LOCK.read_text()


def test_build_writes_the_lock_when_requirements_are_missing(tmp_path: Path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()

    _write_locked_requirements(app_dir)

    assert (app_dir / "requirements.txt").read_text() == RUNTIME_LOCK.read_text()


def test_every_pip_install_becomes_the_one_hashed_lock_install(tmp_path: Path):
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt .\n"
        "RUN pip install --upgrade pip && \\\n"
        "    pip install -r requirements.txt\n"
        "RUN python -m pip install fastapi uvicorn\n"
        "COPY . .\n"
    )

    assert _normalise_dockerfile_install(df) is True

    lines = df.read_text().splitlines()
    assert lines == [
        "FROM python:3.12-slim",
        "WORKDIR /app",
        "COPY requirements.txt .",
        _LOCKED_INSTALL,
        "COPY . .",
    ]


def test_system_package_installs_are_removed(tmp_path: Path):
    # apt resolves against today's Debian mirrors. Nothing in the lock needs
    # a system package, so the line has no job except drift.
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y gcc\n"
        f"{_LOCKED_INSTALL}\n"
    )

    _normalise_dockerfile_install(df)

    assert "apt-get" not in df.read_text()
    assert _LOCKED_INSTALL in df.read_text()


def test_install_normalisation_ignores_a_dockerfile_without_installs(tmp_path: Path):
    # False-positive guard: lines that merely mention pip are not installs.
    df = tmp_path / "Dockerfile"
    original = "FROM python:3.12-slim\nENV PIP_NO_CACHE_DIR=1\nCOPY . .\n"
    df.write_text(original)

    assert _normalise_dockerfile_install(df) is False
    assert df.read_text() == original


def test_build_installs_the_lock_even_when_it_writes_the_dockerfile(tmp_path: Path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "main.py").write_text("# app")

    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ContainerRuntime().build("fallback_env", app_dir)

    dockerfile = (app_dir / "Dockerfile").read_text()
    assert _LOCKED_INSTALL in dockerfile
    assert dockerfile.count("pip install") == 1
    assert (app_dir / "requirements.txt").read_text() == RUNTIME_LOCK.read_text()


def test_build_normalises_requirements_alongside_dockerfile(tmp_path: Path):
    """The full build path must lock BOTH the Dockerfile's install and the
    requirements before docker build runs."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text(
        "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\n"
        "RUN pip install -r requirements.txt\nEXPOSE 8000\n"
        'CMD ["uvicorn", "main:app", "--port", "8000"]\n'
    )
    (app_dir / "requirements.txt").write_text("fastapi\nuvicorn[standard]\n")

    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ContainerRuntime().build("baseline_env", app_dir)

    assert (app_dir / "requirements.txt").read_text() == RUNTIME_LOCK.read_text()
    assert _LOCKED_INSTALL in (app_dir / "Dockerfile").read_text()


# ---------------------------------------------------------------------------
# Base images: pinned by digest, premade images included
# ---------------------------------------------------------------------------

_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")


def test_every_standard_base_image_is_pinned_by_digest():
    for image in STANDARD_BASE_IMAGES:
        assert _DIGEST_RE.search(image), f"{image} can move under its tag"


@pytest.mark.parametrize("template", ["gmail", "slack"])
def test_premade_images_build_from_the_pinned_base_and_the_lock(template):
    premade = Path(__file__).resolve().parents[2] / "docker" / "premade" / template
    dockerfile = (premade / "Dockerfile").read_text()

    assert dockerfile.startswith(f"FROM {FORGE_PYTHON_BASE}\n")
    assert _LOCKED_INSTALL in dockerfile
    assert (premade / "requirements.txt").read_text() == RUNTIME_LOCK.read_text()


@pytest.mark.parametrize("template", ["gmail", "slack"])
def test_building_a_premade_image_does_not_rewrite_its_committed_files(template, tmp_path):
    # Premade images build in place from the repo, so every normaliser must
    # already be satisfied by the committed files, or each build dirties git.
    import shutil

    premade = Path(__file__).resolve().parents[2] / "docker" / "premade" / template
    work = tmp_path / template
    shutil.copytree(premade, work)

    dockerfile = work / "Dockerfile"
    assert _normalise_dockerfile_base(dockerfile) is False
    assert _normalise_dockerfile_port(dockerfile) is False
    assert _normalise_dockerfile_install(dockerfile) is False
    assert _write_locked_requirements(work) is False


def test_mirror_pull_of_a_digest_ref_tags_the_name_without_the_digest():
    # `docker tag` refuses a digest target. Tagging the name alone is enough:
    # the digest ref then resolves to the same content-addressed image.
    image = "python:3.12-slim@sha256:" + "a" * 64

    def pull_side_effect(ref, **_):
        if ref.startswith("public.ecr.aws"):
            return None
        raise RuntimeError(f"failed: {ref}")

    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry", side_effect=pull_side_effect), \
         patch("forge.envgen.container._docker_tag") as mock_tag:
        pull_image(image)

    mock_tag.assert_called_once_with(
        "public.ecr.aws/docker/library/" + image, "python:3.12-slim"
    )


def test_a_digest_ref_never_falls_back_to_the_rebuilt_http_image():
    # The HTTPS loader rebuilds the manifest locally, so its digest can never
    # match the pin. Serving it would silently unpin the build.
    image = "python:3.12-slim@sha256:" + "b" * 64
    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry", side_effect=RuntimeError("EOF")), \
         patch("forge.envgen._image_pull_http.pull_via_http") as mock_http:
        with pytest.raises(RuntimeError, match="digest"):
            pull_image(image)

    mock_http.assert_not_called()


def test_build_normalises_port_alongside_base(tmp_path: Path):
    """build() must run BOTH normalisations before pulling/building, so a
    LLM Dockerfile with FROM python:3.11-slim + EXPOSE 5000 ends up with
    canonical FROM and EXPOSE 8000."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "EXPOSE 5000\n"
        'CMD ["uvicorn", "main:app", "--port", "5000"]\n'
    )

    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ContainerRuntime().build("port_norm_env", app_dir)

    text = (app_dir / "Dockerfile").read_text()
    assert text.startswith(f"FROM {FORGE_PYTHON_BASE}")
    assert f"EXPOSE {FORGE_APP_PORT}" in text
    assert "5000" not in text


def test_build_normalises_dockerfile_before_pull(tmp_path: Path):
    """build() must rewrite python:3.11-slim → canonical, then pull canonical (not 3.11)."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM python:3.11-slim\nWORKDIR /app\n")

    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull, \
         patch("forge.envgen.container.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ContainerRuntime().build("my_env", app_dir)

    # Pull was called with the canonical base, NOT the LLM-chosen 3.11-slim
    mock_pull.assert_called_once_with(FORGE_PYTHON_BASE)
    # And the Dockerfile on disk now points at the canonical base
    assert (app_dir / "Dockerfile").read_text().startswith(f"FROM {FORGE_PYTHON_BASE}")


# ---------------------------------------------------------------------------
# prewarm_standard_base_images — Hub-flakiness defence at worker boot
# ---------------------------------------------------------------------------

def test_prewarm_skips_images_already_cached():
    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull:
        results = prewarm_standard_base_images(images=("python:3.12-slim", "ubuntu:22.04"))

    mock_pull.assert_not_called()
    assert results == {"python:3.12-slim": "cached", "ubuntu:22.04": "cached"}


def test_prewarm_pulls_missing_images():
    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull:
        results = prewarm_standard_base_images(images=("python:3.12-slim", "ubuntu:22.04"))

    assert mock_pull.call_count == 2
    assert results == {"python:3.12-slim": "pulled", "ubuntu:22.04": "pulled"}


def test_prewarm_does_not_raise_when_pull_fails():
    """If every transport is down at boot (registries + direct HTTPS),
    prewarm should log + continue, not crash the worker."""
    def cache_check(image):
        return False

    def pull_fail(image, **_):
        raise RuntimeError(f"Failed to pull {image} after 5 attempts: EOF")

    with patch("forge.envgen.container._image_cached_locally", side_effect=cache_check), \
         patch("forge.envgen.container._pull_with_retry", side_effect=pull_fail), \
         patch("forge.envgen._image_pull_http.pull_via_http",
               side_effect=RuntimeError("HTTPS also EOF")):
        results = prewarm_standard_base_images(images=("python:3.12-slim",))

    assert "python:3.12-slim" in results
    assert results["python:3.12-slim"].startswith("failed:")


def test_prewarm_default_includes_canonical_python_base():
    """The standard set must include FORGE_PYTHON_BASE — otherwise the whole
    point of the pre-warm (eliminating Hub from the build hot path) collapses."""
    assert FORGE_PYTHON_BASE in STANDARD_BASE_IMAGES


# ---------------------------------------------------------------------------
# _is_hub_image — only docker.io references should attempt mirror fallback
# ---------------------------------------------------------------------------

def test_is_hub_image_bare_repo():
    assert _is_hub_image("python:3.12-slim") is True


def test_is_hub_image_user_repo():
    assert _is_hub_image("nginx:latest") is True
    assert _is_hub_image("user/repo:tag") is True


def test_is_hub_image_explicit_registry():
    """A registry like 'lscr.io/...' or 'gcr.io/...' is NOT Docker Hub."""
    assert _is_hub_image("lscr.io/linuxserver/chromium:latest") is False
    assert _is_hub_image("public.ecr.aws/docker/library/python:3.12-slim") is False
    assert _is_hub_image("ghcr.io/user/image:tag") is False


def test_is_hub_image_localhost():
    assert _is_hub_image("localhost:5000/myimage:tag") is False


# ---------------------------------------------------------------------------
# _mirror_ref_for — rewrites Hub references onto a mirror namespace
# ---------------------------------------------------------------------------

def test_mirror_ref_for_bare_repo_uses_library():
    """Bare 'python:3.12-slim' is implicitly 'library/python:3.12-slim' on Hub."""
    assert _mirror_ref_for("python:3.12-slim", "public.ecr.aws/docker") \
        == "public.ecr.aws/docker/library/python:3.12-slim"


def test_mirror_ref_for_user_repo_does_not_add_library():
    """'user/repo:tag' already has a namespace — don't insert 'library'."""
    assert _mirror_ref_for("user/repo:tag", "mirror.gcr.io") \
        == "mirror.gcr.io/user/repo:tag"


# ---------------------------------------------------------------------------
# pull_image — Hub-mirror fallback (the actual fix for Hub EOF outages)
# ---------------------------------------------------------------------------

def test_pull_image_skips_when_already_cached():
    """No network calls when the image is already in the local cache."""
    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull:
        pull_image("python:3.12-slim")

    mock_pull.assert_not_called()


def test_pull_image_uses_canonical_first():
    """Happy path: docker.io works, so no mirrors are tried."""
    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull:
        pull_image("python:3.12-slim")

    mock_pull.assert_called_once_with("python:3.12-slim")


def test_pull_image_falls_back_to_aws_ecr_when_dockerhub_fails(caplog):
    """When docker.io throws EOF, the next mirror (public.ecr.aws) is tried."""
    caplog.set_level(logging.INFO, logger="forge.envgen.container")
    canonical_fail = RuntimeError("Failed to pull python:3.12-slim: EOF")

    def pull_side_effect(ref, **_):
        if ref == "python:3.12-slim":
            raise canonical_fail
        # mirror succeeds
        return None

    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry", side_effect=pull_side_effect) as mock_pull, \
         patch("forge.envgen.container.subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0)
        pull_image("python:3.12-slim")

    assert "docker.io unavailable" in caplog.text
    assert not any(
        record.levelno >= logging.WARNING and "docker.io" in record.message
        for record in caplog.records
    )

    refs_pulled = [c.args[0] for c in mock_pull.call_args_list]
    assert refs_pulled[0] == "python:3.12-slim"
    assert refs_pulled[1] == "public.ecr.aws/docker/library/python:3.12-slim"
    # And the mirror image was retagged as the canonical name so callers stay blind
    tag_calls = [c for c in mock_subproc.call_args_list if c.args[0][:2] == ["docker", "tag"]]
    assert len(tag_calls) == 1
    assert tag_calls[0].args[0] == [
        "docker", "tag",
        "public.ecr.aws/docker/library/python:3.12-slim",
        "python:3.12-slim",
    ]


def test_pull_image_falls_through_to_gcr_when_first_mirror_fails():
    """If both docker.io and AWS ECR fail, mirror.gcr.io is the last resort."""
    def pull_side_effect(ref, **_):
        if ref.startswith("mirror.gcr.io"):
            return None  # GCR mirror succeeds
        raise RuntimeError(f"failed: {ref}")

    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry", side_effect=pull_side_effect) as mock_pull, \
         patch("forge.envgen.container.subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0)
        pull_image("python:3.12-slim")

    refs_pulled = [c.args[0] for c in mock_pull.call_args_list]
    assert refs_pulled == [
        "python:3.12-slim",
        "public.ecr.aws/docker/library/python:3.12-slim",
        "mirror.gcr.io/library/python:3.12-slim",
    ]


def test_pull_image_raises_when_all_registries_and_https_fail():
    """If every registry AND the direct-HTTPS fallback fail, the chain is surfaced."""
    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry", side_effect=RuntimeError("EOF")), \
         patch("forge.envgen._image_pull_http.pull_via_http",
               side_effect=RuntimeError("https EOF too")):
        with pytest.raises(RuntimeError, match="from docker.io, any mirror, or direct HTTPS"):
            pull_image("python:3.12-slim")


def test_pull_image_falls_through_to_direct_https_when_all_docker_pulls_fail():
    """When dockerd's pull pipeline is broken on all registries, the
    Python-native HTTPS pull is tried as a final transport.

    This is the key fix for environments where the Docker daemon's HTTP/2
    client is unstable (MTU mismatch, broken IPv6, idle-stream resets) —
    httpx talks plain HTTP/1.1 over a fresh stack and bypasses dockerd."""
    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry",
               side_effect=RuntimeError("EOF on every docker pull")) as mock_pull, \
         patch("forge.envgen._image_pull_http.pull_via_http") as mock_http:
        # All 3 docker pulls fail; HTTPS succeeds.
        pull_image("python:3.12-slim")

    # Tried docker.io + 2 mirrors via _pull_with_retry, then the HTTPS path
    assert mock_pull.call_count == 3
    mock_http.assert_called_once_with("python:3.12-slim")


def test_pull_image_does_not_use_mirrors_for_non_hub_image():
    """Non-Hub images (lscr.io/...) have no Hub-mirror equivalent — don't try."""
    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry") as mock_pull:
        pull_image("lscr.io/linuxserver/chromium:latest")

    mock_pull.assert_called_once_with("lscr.io/linuxserver/chromium:latest")


def test_pull_image_propagates_failure_for_non_hub_image():
    """Non-Hub image failures bubble up as-is (no mirror fallback applies)."""
    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry", side_effect=RuntimeError("auth denied")):
        with pytest.raises(RuntimeError, match="auth denied"):
            pull_image("lscr.io/linuxserver/chromium:latest")


# ---------------------------------------------------------------------------
# ContainerRuntime.start — defensive against stale-state envs
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_daemon():
    daemon = FakeDocker()
    with patch("forge.envgen.container.docker.from_env", return_value=daemon), \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        yield daemon


def _gateway_port(daemon: FakeDocker, env_name: str) -> int:
    return int(daemon.named(f"forge-{env_name}-gw").ports["8000/tcp"][0]["HostPort"])


def test_start_restarts_existing_stopped_container(fake_daemon):
    """Happy path: the pair exists and the gateway still publishes. Restart it."""
    runtime = ContainerRuntime()
    app_id, _ = runtime.run("my_env", "forge-env-my-env:latest")
    runtime.stop(app_id)

    cid, port = runtime.start("my_env", app_id, "forge-env-my-env:latest")

    assert fake_daemon.containers.get(app_id).status == "running"
    assert (cid, port) == (app_id, _gateway_port(fake_daemon, "my_env"))


def test_start_with_empty_container_id_runs_fresh(fake_daemon):
    """Empty container_id (DB has stale/missing reference) → run a fresh pair."""
    cid, port = ContainerRuntime().start("my_env", "", "forge-env-my-env:latest")

    assert fake_daemon.containers.get(cid).name == "forge-my_env"
    assert port == _gateway_port(fake_daemon, "my_env")


def test_start_removes_stale_named_container_before_running_fresh(fake_daemon):
    """If the old container_id is gone but a same-name twin still exists,
    run() would raise 409 Conflict. start must remove it first."""
    fake_daemon.containers.run(name="forge-my_env", image="old", labels={"forge.env": "my_env"})
    fake_daemon.containers.get("forge-my_env").stop()

    cid, _port = ContainerRuntime().start("my_env", "stale-id", "forge-env-my-env:latest")

    assert "forge-my_env" in fake_daemon.removed
    assert fake_daemon.containers.get(cid).kwargs["image"] == "forge-env-my-env:latest"


def test_start_raises_clear_error_when_image_missing(fake_daemon):
    """If the image was pruned, start() must raise a RuntimeError telling
    the user to rebuild, not a confusing docker.errors.ImageNotFound."""
    fake_daemon.missing_images.add("forge-env-my-env:latest")

    with pytest.raises(RuntimeError, match="must be rebuilt"):
        ContainerRuntime().start("my_env", "old-id", "forge-env-my-env:latest")


def test_start_falls_through_when_existing_container_lost_its_port(fake_daemon):
    """An app whose gateway is gone (host reboot, or started before gateways)
    is dropped and replaced by a fresh pair with a working port."""
    runtime = ContainerRuntime()
    old_id, _ = runtime.run("my_env", "forge-env-my-env:latest")
    fake_daemon.named("forge-my_env-gw").remove(force=True)

    cid, port = runtime.start("my_env", old_id, "forge-env-my-env:latest")

    assert cid != old_id
    assert port == _gateway_port(fake_daemon, "my_env")


def test_start_falls_through_when_existing_container_refuses_to_start(fake_daemon):
    """Container exists but .start() raises (image vanished etc.) → remove + fresh."""
    runtime = ContainerRuntime()
    broken_id, _ = runtime.run("my_env", "forge-env-my-env:latest")
    fake_daemon.refuse_start.add("forge-my_env")

    try:
        cid, _port = runtime.start("my_env", broken_id, "forge-env-my-env:latest")
    finally:
        fake_daemon.refuse_start.clear()

    assert cid != broken_id
    assert fake_daemon.containers.get(cid).name == "forge-my_env"


# ---------------------------------------------------------------------------
# _wait_for_port_binding — polls Docker for the host-port mapping
# ---------------------------------------------------------------------------

def test_wait_for_port_binding_returns_port_on_first_try():
    container = MagicMock()
    container.ports = {"8000/tcp": [{"HostPort": "32777"}]}
    with patch("forge.envgen.container.time.sleep") as mock_sleep:
        port = _wait_for_port_binding(container, "8000/tcp", attempts=5, interval=0.1)
    assert port == 32777
    mock_sleep.assert_not_called()


def test_wait_for_port_binding_polls_until_binding_appears():
    """Daemon may not have allocated the host port at the moment of run() —
    the helper must keep reloading until the binding shows up."""
    container = MagicMock()
    # First two reloads: no binding. Third: binding present.
    states = [
        {},
        {"8000/tcp": []},
        {"8000/tcp": [{"HostPort": "32100"}]},
    ]
    iteration = {"i": 0}

    def reload_side_effect():
        container.ports = states[iteration["i"]]
        iteration["i"] += 1

    container.reload.side_effect = reload_side_effect

    with patch("forge.envgen.container.time.sleep"):
        port = _wait_for_port_binding(container, "8000/tcp", attempts=5, interval=0.1)
    assert port == 32100
    assert container.reload.call_count == 3


def test_wait_for_port_binding_raises_with_diagnostic_when_never_appears():
    """If the binding never appears, raise a clear RuntimeError that names
    the port and shows what container.ports actually held."""
    container = MagicMock()
    container.id = "abcdef1234567890"
    container.ports = {}  # never gets a binding

    with patch("forge.envgen.container.time.sleep"):
        with pytest.raises(RuntimeError, match=r"never reported a host-port binding for 8000/tcp"):
            _wait_for_port_binding(container, "8000/tcp", attempts=3, interval=0.01)


def test_wait_for_port_binding_skips_zero_or_invalid_host_port():
    """A binding that says HostPort='0' or empty string is not a real port —
    the helper must keep waiting rather than returning 0."""
    container = MagicMock()
    states = [
        {"8000/tcp": [{"HostPort": ""}]},
        {"8000/tcp": [{"HostPort": "0"}]},
        {"8000/tcp": [{"HostPort": "32200"}]},
    ]
    iteration = {"i": 0}

    def reload_side_effect():
        container.ports = states[iteration["i"]]
        iteration["i"] += 1

    container.reload.side_effect = reload_side_effect

    with patch("forge.envgen.container.time.sleep"):
        port = _wait_for_port_binding(container, "8000/tcp", attempts=5, interval=0.01)
    assert port == 32200


def test_hub_mirrors_constant_is_in_priority_order():
    """AWS Public ECR is checked before Google's mirror — both are reliable
    but ECR has historically had better Hub-image freshness."""
    assert _HUB_MIRRORS == ("public.ecr.aws/docker", "mirror.gcr.io")


def test_build_gives_up_on_a_hung_docker_build(tmp_path):
    """A stuck build must free the worker slot instead of hanging forever."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")

    with patch("forge.envgen.container.subprocess.run") as mock_run, \
         patch("forge.envgen.container._image_cached_locally", return_value=True):
        mock_run.side_effect = subprocess.TimeoutExpired(["docker", "build"], 900)
        with pytest.raises(RuntimeError, match="docker build timed out after 900s"):
            ContainerRuntime().build("hung_env", app_dir)

    assert mock_run.call_args.kwargs["timeout"] == 900


# ---------------------------------------------------------------------------
# Launch-time determinism: network and timezone
# ---------------------------------------------------------------------------

def _launch_kwargs(launch) -> dict:
    """Run one ContainerRuntime launch on a fake daemon and return the env container's kwargs."""
    daemon = FakeDocker()
    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container.docker.from_env", return_value=daemon):
        launch(ContainerRuntime())
    env_containers = [
        c for c in daemon.containers._by_id.values() if c.labels.get("forge.role") != "gateway"
    ]
    return env_containers[0].kwargs


def test_cli_container_has_no_network():
    # The CLI agent drives the shell through `docker exec`, so the container
    # needs no network at all, and a live one lets `curl`/`apt` see today's
    # internet instead of a fixed world.
    kwargs = _launch_kwargs(lambda rt: rt.run_cli("shell_env"))

    assert kwargs["network_mode"] == "none"




@pytest.mark.parametrize("launch", [
    lambda rt: rt.run("app_env", "forge-env-app-env:latest"),
    lambda rt: rt.run_cli("shell_env"),
    lambda rt: rt.run_browser("browser_env"),
])
def test_every_container_runs_in_utc(launch):
    # The host's timezone must not leak into `date`, log lines, or any local
    # time an app renders, or the same episode differs between machines.
    kwargs = _launch_kwargs(launch)

    assert kwargs["environment"]["TZ"] == "UTC"


@pytest.mark.parametrize("launch", [
    lambda rt: rt.run("app_env", "forge-env-app-env:latest"),
    lambda rt: rt.run_cli("shell_env"),
])
def test_python_in_a_container_iterates_sets_in_a_fixed_order(launch):
    # Python salts str hashes per process, so a set of names iterates in a
    # different order after every container start. Code that builds state
    # from a set would then differ run to run. A fixed seed removes the salt.
    kwargs = _launch_kwargs(launch)

    assert kwargs["environment"]["PYTHONHASHSEED"] == "0"


# ---------------------------------------------------------------------------
# CLI clock: a prebuilt image with libfaketime freezes what commands see
# ---------------------------------------------------------------------------

def test_cli_commands_see_the_fixed_epoch_not_the_host_clock():
    # Every process in the shell starts at the SimClock epoch, so `date`,
    # `python -c 'import time'`, and log lines read the same time every run.
    kwargs = _launch_kwargs(lambda rt: rt.run_cli("shell_env"))

    assert kwargs["image"] == CLI_RUNTIME_IMAGE
    assert kwargs["environment"]["LD_PRELOAD"] == "/usr/local/lib/libfaketime.so.1"
    assert kwargs["environment"]["FAKETIME"] == "@2023-11-14 22:13:20"


def test_cli_image_is_built_from_the_pinned_base_when_missing():
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry"), \
         patch("forge.envgen.container.subprocess.run", side_effect=fake_run):
        assert ensure_cli_image() == CLI_RUNTIME_IMAGE

    builds = [c for c in calls if c[:2] == ["docker", "build"]]
    assert len(builds) == 1
    assert f"BASE_IMAGE={FORGE_CLI_IMAGE}" in builds[0]
    assert ["-t", CLI_RUNTIME_IMAGE] == builds[0][builds[0].index("-t"):builds[0].index("-t") + 2]


def test_cli_image_is_not_rebuilt_when_already_cached():
    with patch("forge.envgen.container._image_cached_locally", return_value=True), \
         patch("forge.envgen.container.subprocess.run") as mock_run:
        assert ensure_cli_image() == CLI_RUNTIME_IMAGE

    mock_run.assert_not_called()


def test_a_failed_cli_image_build_raises_with_dockers_output():
    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["docker", "build"]:
            raise subprocess.CalledProcessError(1, cmd, stderr="E: Unable to locate package")
        return MagicMock(returncode=0)

    with patch("forge.envgen.container._image_cached_locally", return_value=False), \
         patch("forge.envgen.container._pull_with_retry"), \
         patch("forge.envgen.container.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="Unable to locate package"):
            ensure_cli_image()


def test_cli_image_tag_follows_the_dockerfile_content():
    # Editing the Dockerfile must produce a new tag, or a stale image is reused.
    assert CLI_RUNTIME_IMAGE.startswith("forge-cli:")
    assert CLI_RUNTIME_IMAGE != _cli_image_tag(CLI_DOCKERFILE.read_text() + "\n# edit")
