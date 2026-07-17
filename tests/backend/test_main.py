from __future__ import annotations

import logging
from unittest.mock import patch

import docker.errors

from backend.app.main import _reattach_containers


def test_reattach_treats_missing_docker_socket_as_expected(caplog):
    error = docker.errors.DockerException(
        "Error while fetching server API version",
        ("Connection aborted", FileNotFoundError(2, "No such file or directory")),
    )

    with (
        caplog.at_level(logging.INFO, logger="backend.app.main"),
        patch("forge.envgen.container.ContainerRuntime", side_effect=error),
    ):
        _reattach_containers()

    assert "Docker is not running; container reattach disabled" in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_reattach_keeps_unexpected_failures_visible(caplog):
    with (
        caplog.at_level(logging.WARNING, logger="backend.app.main"),
        patch(
            "forge.envgen.container.ContainerRuntime",
            side_effect=RuntimeError("unexpected reattach failure"),
        ),
    ):
        _reattach_containers()

    assert "container reattach failed" in caplog.text
    assert any(record.levelno == logging.WARNING for record in caplog.records)
