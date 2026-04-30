from __future__ import annotations
import os
import re
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

    @staticmethod
    def _container_name(env_name: str) -> str:
        """Return a Docker-safe container name for an environment."""
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", env_name)
        return f"forge-{safe}"

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

    def _remove_existing(self, env_name: str) -> None:
        """Remove an existing container with the forge-<env_name> name if present."""
        try:
            old = self._docker.containers.get(self._container_name(env_name))
            old.remove(force=True)
        except docker.errors.NotFound:
            pass

    def run_cli(self, env_name: str) -> tuple[str, int]:
        """Spin up an Ubuntu 22.04 shell container (no HTTP port)."""
        # Pull via CLI to avoid credential-helper hangs (same reason as build())
        subprocess.run(
            ["docker", "pull", "ubuntu:22.04"],
            check=True,
            capture_output=True,
            text=True,
        )
        self._remove_existing(env_name)
        container = self._docker.containers.run(
            image="ubuntu:22.04",
            name=self._container_name(env_name),
            command=["tail", "-f", "/dev/null"],
            detach=True,
            labels={"forge.env": env_name, "forge.managed": "true", "forge.type": "cli"},
            restart_policy={"Name": "unless-stopped"},
        )
        return container.id, 0

    def run_browser(self, env_name: str) -> tuple[str, int]:
        """Spin up a Chromium+KasmVNC container, exposing the web UI on a random port."""
        # Pull via CLI to avoid credential-helper hangs (same reason as build())
        subprocess.run(
            ["docker", "pull", "lscr.io/linuxserver/chromium:latest"],
            check=True,
            capture_output=True,
            text=True,
        )
        self._remove_existing(env_name)
        container = self._docker.containers.run(
            image="lscr.io/linuxserver/chromium:latest",
            name=self._container_name(env_name),
            detach=True,
            ports={"3001/tcp": None},
            environment={
                "CUSTOM_USER": "forge",
                "PASSWORD": "forge",
                "PUID": "1000",
                "PGID": "1000",
                "TZ": "UTC",
            },
            shm_size="1g",
            labels={"forge.env": env_name, "forge.managed": "true", "forge.type": "browser"},
            restart_policy={"Name": "unless-stopped"},
        )
        container.reload()
        port = int(container.ports["3001/tcp"][0]["HostPort"])
        return container.id, port

    def run(self, env_name: str, image_tag: str) -> tuple[str, int]:
        container = self._docker.containers.run(
            image=image_tag,
            name=self._container_name(env_name),
            detach=True,
            ports={"8000/tcp": None},
            environment={"REDIS_URL": self._redis_url, "FORGE_ENV_NAME": env_name},
            labels={"forge.env": env_name, "forge.managed": "true"},
            restart_policy={"Name": "unless-stopped"},
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

    def start(self, env_name: str, container_id: str, image_tag: str) -> tuple[str, int]:
        """Restart a stopped container, or run a fresh one if it was removed."""
        try:
            container = self._docker.containers.get(container_id)
            container.start()
            container.reload()
            if image_tag == "builtin:cli":
                return container.id, 0
            elif image_tag == "builtin:browser":
                port = int(container.ports["3001/tcp"][0]["HostPort"])
                return container.id, port
            else:
                port = int(container.ports["8000/tcp"][0]["HostPort"])
                return container.id, port
        except docker.errors.NotFound:
            if image_tag == "builtin:cli":
                return self.run_cli(env_name)
            elif image_tag == "builtin:browser":
                return self.run_browser(env_name)
            else:
                return self.run(env_name, image_tag)

    def remove(self, container_id: str, image_tag: str | None = None) -> None:
        self.stop(container_id)
        try:
            container = self._docker.containers.get(container_id)
            container.remove()
        except docker.errors.NotFound:
            pass
        # Only remove custom-built images; never touch shared builtin images
        if image_tag and not image_tag.startswith("builtin:"):
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
