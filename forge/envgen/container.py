from __future__ import annotations
import os
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
        self._docker = docker.from_env()
        self._redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    def build(self, env_name: str, app_dir: Path) -> str:
        dockerfile = app_dir / "Dockerfile"
        if not dockerfile.exists():
            dockerfile.write_text(_DOCKERFILE)
        image, _ = self._docker.images.build(
            path=str(app_dir),
            tag=f"forge-env:{env_name}:latest",
            rm=True,
        )
        return image.tags[0]

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
