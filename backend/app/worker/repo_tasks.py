"""Celery task that turns one approved GitHub repo into a running repo environment."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from backend.app.worker.celery_app import celery

logger = logging.getLogger(__name__)


def production_build_deps():
    """The real download, safety gate, Docker build, and launcher, configured from env."""
    import docker

    from backend.app.services.repo_pipeline import RepoBuildDeps
    from forge.envgen import docker_images
    from forge.envgen.container import ContainerRuntime
    from forge.envgen.docker_cli import SubprocessDockerCLI
    from forge.repogen.build import DockerTestSandbox, build_repo_environment
    from forge.repogen.config import repo_limits_from_env
    from forge.repogen.github_mcp import github_settings_from_env
    from forge.repogen.snapshot import fetch_tarball, safe_extract
    from forge.safety.config import safety_service_from_env
    from forge.safety.types import DataSource

    _, token = github_settings_from_env()
    limits = repo_limits_from_env()
    # Code keeps names: the name model would rewrite identifiers like package names.
    safety = safety_service_from_env(DataSource.INTERNET, detect_names=False).service

    def download(full_name: str, sha: str, work_dir: Path) -> Path:
        tarball = fetch_tarball(full_name, sha, token, work_dir / "repo.tgz", max_bytes=limits.max_tarball_bytes)
        # Source compresses about 5:1, so a larger expansion is refused as a bomb.
        return safe_extract(tarball, work_dir / "src", max_bytes=limits.max_tarball_bytes * 5)

    def build(env_name: str, root: Path, language: str | None):
        return build_repo_environment(
            env_name, root, language,
            docker=SubprocessDockerCLI(), sandbox=DockerTestSandbox(env_name, docker.from_env()),
            min_stable_tests=limits.min_stable_tests, pull=docker_images.pull_image,
        )

    def launch(env_name: str, image_tag: str) -> str:
        return ContainerRuntime().run_repo(env_name, image_tag)[0]

    return RepoBuildDeps(download=download, safety=safety, build=build, launch=launch,
                         work_root=Path(tempfile.gettempdir()))


@celery.task(name="backend.app.worker.tasks.build_repo_task", ignore_result=True)
def build_repo_task(env_name: str) -> None:
    from backend.app.database import get_session_factory
    from backend.app.services.repo_pipeline import build_repo, fail_build
    from forge.repogen.github_mcp import RepoConfigError
    from forge.safety.config import SafetyConfigError

    with get_session_factory()() as db:
        try:
            deps = production_build_deps()
        except (RepoConfigError, SafetyConfigError) as exc:
            logger.error("repo build for %s cannot start: %s", env_name, exc)
            fail_build(db, env_name, str(exc))
            return
        env = build_repo(db, env_name, deps)
        logger.info("repo build for %s finished: %s %s", env_name, env.status, env.reason or "")
