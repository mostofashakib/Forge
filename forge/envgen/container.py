from __future__ import annotations
import os
import subprocess
from pathlib import Path
import docker
import docker.errors


_DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn sqlalchemy redis httpx
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""


class ContainerRuntime:
    def __init__(self) -> None:
        self._docker_client: docker.DockerClient | None = None
        self._redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    @property
    def _docker(self) -> docker.DockerClient:
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    def build(self, env_name: str, app_dir: Path) -> str:
        dockerfile = app_dir / "Dockerfile"
        if not dockerfile.exists():
            dockerfile.write_text(_DOCKERFILE)
        safe_name = env_name.replace("_", "-").lower()
        tag = f"forge-env-{safe_name}:latest"
        # Use CLI directly — the Python SDK credential-helper resolution fails
        # when optional helpers (e.g. docker-credential-gcloud) are configured
        # but not authenticated, even for Docker Hub images.
        subprocess.run(
            ["docker", "build", "-t", tag, "--rm", str(app_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return tag

    def run(self, env_name: str, image_tag: str) -> tuple[str, int]:
        container = self._docker.containers.run(
            image=image_tag,
            name=f"forge-{env_name}",
            detach=True,
            ports={"8000/tcp": None},
            environment={"REDIS_URL": self._redis_url, "FORGE_ENV_NAME": env_name},
            labels={"forge.env": env_name, "forge.managed": "true"},
        )
        container.reload()
        port = int(container.ports["8000/tcp"][0]["HostPort"])
        return container.id, port

    def stop(self, container_id: str) -> None:
        try:
            container = self._docker.containers.get(container_id)
            container.stop(timeout=10)
        except docker.errors.NotFound:
            pass

    def remove(self, container_id: str, image_tag: str | None = None) -> None:
        self.stop(container_id)
        try:
            container = self._docker.containers.get(container_id)
            container.remove()
        except docker.errors.NotFound:
            pass
        if image_tag:
            try:
                self._docker.images.remove(image_tag, force=True)
            except docker.errors.ImageNotFound:
                pass

    def reattach_all(self) -> list[tuple[str, str, int]]:
        containers = self._docker.containers.list(
            filters={"label": "forge.managed=true"}
        )
        result = []
        for c in containers:
            env_name = c.labels.get("forge.env", "")
            c.reload()
            ports = c.ports.get("8000/tcp")
            if env_name and ports:
                result.append((env_name, c.id, int(ports[0]["HostPort"])))
        return result
