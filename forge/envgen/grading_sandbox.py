"""Run CLI grading assertions in a fresh copy of the agent's container, out of its reach.

CLI grading uses only a toolbox: a static BusyBox from a pinned image,
copied into a volume and mounted read-only into the grader. Static binaries
ignore LD_PRELOAD and /etc/ld.so.preload, so nothing the agent wrote in its
container can change what an assertion's `grep` or `test` reports.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager

from forge.envgen.docker_cli import DockerCLI, SubprocessDockerCLI
from forge.envgen.docker_images import FORGE_GRADER_TOOLBOX_IMAGE, pull_image

_GRADER_TOOLBOX_DIR = "/opt/forge-grader"
_GRADER_PATH = (
    f"{_GRADER_TOOLBOX_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


def _grader_toolbox_volume() -> str:
    digest = hashlib.sha256(FORGE_GRADER_TOOLBOX_IMAGE.encode()).hexdigest()[:12]
    return f"forge-grader-tools-{digest}"


def ensure_grader_toolbox(cli: DockerCLI | None = None) -> str:
    """Return the read-only grader toolbox volume, populating it on first use.

    The volume only exists once it is fully populated, so its existence is
    the readiness signal.
    """
    cli = cli or SubprocessDockerCLI()
    volume = _grader_toolbox_volume()
    if cli.succeeds("volume", "inspect", volume):
        return volume
    pull_image(FORGE_GRADER_TOOLBOX_IMAGE)
    cli.run("volume", "create", "--label", "forge.managed=true", volume)
    try:
        # Installed at the path the grader mounts it on, so the applet
        # symlinks stay valid there.
        cli.run(
            "run", "--rm", "--network", "none",
            "-v", f"{volume}:{_GRADER_TOOLBOX_DIR}",
            FORGE_GRADER_TOOLBOX_IMAGE, "sh", "-c",
            f"mkdir -p {_GRADER_TOOLBOX_DIR}/bin"
            f" && cp /bin/busybox {_GRADER_TOOLBOX_DIR}/bin/busybox"
            f" && {_GRADER_TOOLBOX_DIR}/bin/busybox --install -s {_GRADER_TOOLBOX_DIR}/bin",
        )
    except RuntimeError as exc:
        cli.discard("volume", "rm", "-f", volume)
        raise RuntimeError(f"Could not prepare the grader toolbox: {exc}") from exc
    return volume


@contextmanager
def grading_sandbox(container_id: str, cli: DockerCLI | None = None) -> Iterator[list[str]]:
    """Yield the command prefix that runs one shell assertion out of the agent's reach.

    The agent's container is committed to an image and a fresh grader starts
    from it: same files, none of the agent's processes, no network. Each
    assertion runs through the read-only toolbox's static shell with a clean
    environment. Append the assertion string to the yielded list.
    """
    cli = cli or SubprocessDockerCLI()
    toolbox = ensure_grader_toolbox(cli)
    name = f"forge-grade-{container_id[:12]}"
    image = f"{name}:snapshot"
    cli.run("commit", container_id, image)
    try:
        cli.run(
            "run", "-d", "--name", name, "--network", "none",
            "--label", "forge.role=grader",
            "-v", f"{toolbox}:{_GRADER_TOOLBOX_DIR}:ro",
            "--entrypoint", f"{_GRADER_TOOLBOX_DIR}/bin/sleep",
            image, "infinity",
        )
        try:
            yield [
                "docker", "exec", name,
                f"{_GRADER_TOOLBOX_DIR}/bin/env", "-i",
                f"PATH={_GRADER_PATH}", "HOME=/root", "TZ=UTC",
                f"{_GRADER_TOOLBOX_DIR}/bin/sh", "-c",
            ]
        finally:
            cli.discard("rm", "-f", name)
    finally:
        cli.discard("rmi", "-f", image)
