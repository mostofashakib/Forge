"""Build a repo environment: image with dependencies, then a two-run test baseline.

Dependencies install with network while the image builds. The baseline then
runs the tests twice, each time in a fresh container with the CLI isolation
every episode gets (no network, frozen clock). Tests that pass both runs are
the stable set tasks may rely on. Any failure drops the repo with its reason.
"""
from __future__ import annotations

import subprocess
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from forge.envgen.docker_cli import DockerCLI
from forge.repogen.toolchains import Toolchain, detect_toolchain

REPO_IMAGE_PREFIX = "forge-repo-"
BASELINE_RUNS = 2
_LOG_CHARS = 4000
_FAKETIME_LIB = "/usr/local/lib/libfaketime.so.1"


class TestSandbox(Protocol):
    def run(self, image: str, command: str, timeout_s: float) -> str:
        """Run `command` in a fresh isolated container from `image` and return its stdout."""


@dataclass(frozen=True)
class RepoBuildResult:
    status: str  # "ready" or "dropped"
    reason: str = ""
    image_tag: str | None = None
    toolchain: str | None = None
    test_command: str | None = None
    stable_tests: list[str] = field(default_factory=list)
    log: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def repo_image_tag(env_name: str) -> str:
    return f"{REPO_IMAGE_PREFIX}{env_name.replace('_', '-').lower()}:latest"


def render_dockerfile(toolchain: Toolchain, root: Path) -> str:
    lines = [
        f"FROM {toolchain.base_image}",
        # A tarball has no .git, which setuptools-scm needs to compute a version.
        "ENV CI=1 PIP_DISABLE_PIP_VERSION_CHECK=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0",
        "RUN apt-get update && apt-get install -y --no-install-recommends libfaketime git"
        " && rm -rf /var/lib/apt/lists/*"
        f" && ln -s \"$(dpkg -L libfaketime | grep '/libfaketime.so.1$')\" {_FAKETIME_LIB}",
        "COPY . /repo",
        "WORKDIR /repo",
        *(f"RUN {command}" for command in toolchain.install_commands(root)),
    ]
    return "\n".join(lines) + "\n"


def build_repo_environment(
    env_name: str,
    root: Path,
    language: str | None,
    *,
    docker: DockerCLI,
    sandbox: TestSandbox,
    min_stable_tests: int,
    pull: Callable[[str], None],
    test_timeout_s: float = 900,
) -> RepoBuildResult:
    toolchain = detect_toolchain(root, language)
    if toolchain is None:
        return RepoBuildResult("dropped", "no supported toolchain")
    tag = repo_image_tag(env_name)
    # Forge's Dockerfile and ignore file sit beside the repo, so the repo's own
    # Dockerfile and .dockerignore (which often excludes tests) are never used.
    dockerfile = root.parent / "Dockerfile.forge"
    dockerfile.write_text(render_dockerfile(toolchain, root))
    (root.parent / "Dockerfile.forge.dockerignore").write_text(".git\n")
    pull(toolchain.base_image)
    try:
        docker.run("build", "-f", str(dockerfile), "-t", tag, "--rm", str(root))
    except RuntimeError as exc:
        return RepoBuildResult("dropped", "install failed", toolchain=toolchain.name, log=str(exc)[-_LOG_CHARS:])

    test_command = toolchain.test_command(toolchain.report_path, root)
    run = f"cd /repo && ({test_command}) >/tmp/forge-test.log 2>&1; cat {toolchain.report_path}"
    try:
        runs = [toolchain.parse_report(sandbox.run(tag, run, test_timeout_s)) for _ in range(BASELINE_RUNS)]
    except (RuntimeError, ValueError) as exc:
        docker.discard("rmi", "-f", tag)
        return RepoBuildResult("dropped", "baseline could not run", toolchain=toolchain.name, log=str(exc)[-_LOG_CHARS:])

    stable = sorted(test for test, passed in runs[0].items() if passed and all(r.get(test) for r in runs[1:]))
    common = dict(toolchain=toolchain.name, test_command=test_command, stable_tests=stable)
    if len(stable) < min_stable_tests:
        docker.discard("rmi", "-f", tag)
        return RepoBuildResult("dropped", f"{len(stable)} stable tests, need {min_stable_tests}", **common)
    return RepoBuildResult("ready", image_tag=tag, **common)


class DockerTestSandbox:
    """Runs a command in a throwaway container with the CLI episode isolation."""

    def __init__(self, env_name: str, docker_client) -> None:
        self._env_name = env_name
        self._docker = docker_client

    def run(self, image: str, command: str, timeout_s: float) -> str:
        from forge.envgen.container_specs import cli_container

        name = f"forge-repo-baseline-{uuid.uuid4().hex[:12]}"
        container = self._docker.containers.run(**cli_container(
            self._env_name, image=image, name=name, labels={"forge.role": "baseline"},
        ))
        try:
            proc = subprocess.run(
                ["docker", "exec", container.id, "bash", "-c", command],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"tests timed out after {timeout_s}s") from exc
        finally:
            container.remove(force=True)
        if not proc.stdout.strip():
            raise RuntimeError(f"the test run wrote no report: {proc.stderr.strip()[-500:]}")
        return proc.stdout
