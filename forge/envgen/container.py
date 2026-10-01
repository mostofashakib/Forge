"""Build, start, stop and remove an environment's containers.

`ContainerRuntime` owns the lifecycle. It composes collaborators that it
receives through its constructor, each with one job:
  - `SandboxNetwork`: the private network and the loopback gateways
  - `EpisodeSnapshots`: per-run CLI forks and per-episode clones
  - `DockerCLI`: every command that goes through the docker binary
Image naming and pulling live in `docker_images`, Dockerfile rewriting in
`dockerfile`, and launch settings in `container_specs`.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import docker
import docker.errors

from forge.envgen import docker_images
from forge.envgen.container_specs import (
    BROWSER_CDP_PORT,
    BROWSER_CDP_RELAY_PORT,
    BROWSER_UI_PORT,
    CPU_LIMIT,
    FORGE_APP_PORT,
    GATEWAY_MEMORY_LIMIT,
    HARDENED,
    PID_LIMIT,
    app_container,
    browser_container,
)
from forge.envgen.docker_cli import DockerCLI, SubprocessDockerCLI
from forge.envgen.dockerfile import (
    DEFAULT_DOCKERFILE,
    normalise_dockerfile_base,
    normalise_dockerfile_install,
    normalise_dockerfile_port,
    parse_from_image,
    write_locked_requirements,
)
from forge.envgen.episode_snapshots import EpisodeSnapshots
from forge.envgen.sandbox_network import (
    SandboxNetwork,
    container_name,
    gateway_name,
    relay_name,
    wait_for_port_binding,
)
from forge.runtime.network_isolation import check_generated_env
from forge.settings import redis_url


class ContainerRuntime:
    def __init__(
        self,
        docker_client: docker.DockerClient | None = None,
        *,
        cli: DockerCLI | None = None,
        network: SandboxNetwork | None = None,
        snapshots: EpisodeSnapshots | None = None,
    ) -> None:
        self._docker_client = docker_client
        self._redis_url = redis_url()
        self._cli = cli or SubprocessDockerCLI()
        self._network = network or SandboxNetwork(self._client)
        self._snapshots = snapshots or EpisodeSnapshots(self._client, self._network, self._cli, self._redis_url)

    def _client(self) -> docker.DockerClient:
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    @property
    def _docker(self) -> docker.DockerClient:
        return self._client()

    def image_exists(self, image: str) -> bool:
        """Return True if image exists in daemon."""
        if self._docker_client is not None:
            try:
                self._docker_client.images.get(image)
                return True
            except Exception:
                return False
        return docker_images.image_cached_locally(image)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, env_name: str, app_dir: Path) -> str:
        dockerfile = app_dir / "Dockerfile"
        if not dockerfile.exists():
            dockerfile.write_text(DEFAULT_DOCKERFILE)
        else:
            # Normalise any LLM-generated `FROM python:X` to the single
            # canonical Forge base. This guarantees every build depends
            # on the one image we pre-warm, so Hub is never on the hot path.
            normalise_dockerfile_base(dockerfile)
            # Force EXPOSE / --port to FORGE_APP_PORT so the in-container
            # listener and the host-side port mapping never disagree, no
            # matter what port the LLM picked.
            normalise_dockerfile_port(dockerfile)
            # Only the hashed lock may be installed, so nothing is resolved
            # at build time.
            normalise_dockerfile_install(dockerfile)

        # The lock replaces whatever the LLM listed. It also covers the
        # runtime deps generated code needs, which the LLM that writes
        # requirements.txt routinely forgets.
        write_locked_requirements(app_dir)

        violations = check_generated_env(app_dir)
        if violations:
            details = ", ".join(
                f"{violation.filename}: {violation.import_line}"
                for violation in violations
            )
            raise RuntimeError(f"Generated environment violates network policy: {details}")

        # Pre-pull the base image before `docker build` to avoid
        # intermittent EOF errors when Docker tries to pull mid-build.
        # `pull_image` skips when the image is already cached (typical case
        # after worker startup pre-warm) and falls back to Hub mirrors
        # (public.ecr.aws, mirror.gcr.io) when docker.io is unreachable.
        base_image = parse_from_image(dockerfile)
        if base_image:
            docker_images.pull_image(base_image)

        safe_name = env_name.replace("_", "-").lower()
        tag = f"forge-env-{safe_name}:latest"
        self._cli.run("build", "-t", tag, "--rm", str(app_dir))
        return tag

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _remove_existing(self, env_name: str) -> None:
        """Remove every container that belongs to an environment, if present."""
        base = container_name(env_name)
        for name in (
            relay_name(env_name), base, gateway_name(env_name),
            f"{base}-warm", f"{base}-episode",
        ):
            try:
                self._docker.containers.get(name).remove(force=True)
            except docker.errors.NotFound:
                pass

    def run_cli(self, env_name: str) -> tuple[str, int]:
        """Spin up an Ubuntu 22.04 shell container (no HTTP port)."""
        image = docker_images.ensure_cli_image()
        self._remove_existing(env_name)
        container = self._snapshots.run_cli_container(env_name, container_name(env_name), image)
        return container.id, 0

    def run_browser(self, env_name: str) -> tuple[str, int]:
        """Spin up a Chromium+KasmVNC container with no route out.

        Its gateway publishes the web UI and the DevTools port on loopback.
        """
        docker_images.pull_image(docker_images.FORGE_BROWSER_IMAGE)
        docker_images.pull_image(docker_images.FORGE_GATEWAY_IMAGE)
        self._remove_existing(env_name)
        network = self._network.ensure(env_name)
        # No Docker init: the image boots through s6-overlay, which must
        # be PID 1 and reaps zombies itself.
        container = self._docker.containers.run(**browser_container(
            env_name,
            image=docker_images.FORGE_BROWSER_IMAGE,
            name=container_name(env_name),
            network=network.name,
            labels={},
            restart_policy={"Name": "unless-stopped"},
        ))
        self._docker.containers.run(
            image=docker_images.FORGE_GATEWAY_IMAGE,
            name=relay_name(env_name),
            entrypoint=["/bin/sh", "-c"],
            command=[
                f"socat TCP-LISTEN:{BROWSER_CDP_RELAY_PORT},fork,reuseaddr "
                f"TCP:127.0.0.1:{BROWSER_CDP_PORT}"
            ],
            detach=True,
            # Inside the browser's namespace: it reaches Chromium's loopback
            # and, like the browser, has no route out.
            network_mode=f"container:{container.id}",
            labels={"forge.env": env_name, "forge.managed": "true", "forge.role": "cdp-relay"},
            restart_policy={"Name": "unless-stopped"},
            mem_limit=GATEWAY_MEMORY_LIMIT,
            nano_cpus=CPU_LIMIT,
            pids_limit=PID_LIMIT,
            **HARDENED,
        )
        gateway = self._network.start_environment_gateway(
            env_name,
            {BROWSER_UI_PORT: BROWSER_UI_PORT, BROWSER_CDP_PORT: BROWSER_CDP_RELAY_PORT},
            network,
        )
        port = wait_for_port_binding(gateway, f"{BROWSER_UI_PORT}/tcp", attempts=10, interval=0.3)
        return container.id, port

    def run(self, env_name: str, image_tag: str) -> tuple[str, int]:
        docker_images.pull_image(docker_images.FORGE_GATEWAY_IMAGE)
        network = self._network.ensure(env_name)
        container = self._docker.containers.run(**app_container(
            env_name,
            image=image_tag,
            name=container_name(env_name),
            # No published port and no route out: only the gateway reaches it.
            network=network.name,
            redis_url=self._redis_url,
            labels={},
            # `on-failure` with a small retry cap, NOT `unless-stopped`:
            # a buggy LLM-generated app that crashes on boot would otherwise
            # restart forever and the UI would oscillate between "running"
            # and "restarting" without surfacing the actual error. With this
            # policy the container exits cleanly after a few crashes so the
            # GET cross-check can flag it and the user can see logs.
            restart_policy={"Name": "on-failure", "MaximumRetryCount": 3},
        ))
        gateway = self._network.start_environment_gateway(env_name, {FORGE_APP_PORT: FORGE_APP_PORT}, network)
        # The port binding is applied asynchronously by the daemon — usually
        # it's there immediately after reload(), but on a busy macOS Docker
        # Desktop it can take a few hundred ms. Poll briefly.
        port = wait_for_port_binding(gateway, f"{FORGE_APP_PORT}/tcp", attempts=10, interval=0.3)
        return container.id, port

    def host_port(self, container, container_port: int | None = None) -> int | None:
        """The loopback port Forge reaches `container_port` of this environment on."""
        return self._network.host_port(container, container_port)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self, container_id: str) -> None:
        try:
            container = self._docker.containers.get(container_id)
        except docker.errors.NotFound:
            return
        container.stop(timeout=10)
        for sidecar in self._network.sidecars_for(container):
            sidecar.stop(timeout=10)

    def start(self, env_name: str, container_id: str, image_tag: str) -> tuple[str, int]:
        """Restart a stopped container, or run a fresh one if it was removed.

        Defensive against the messy states that show up in practice:
          - empty/None container_id (DB reset, partial-state envs) → run fresh
          - container exists but its bound port is gone → run fresh
          - stale stopped container with same forge-<env> name → remove first
          - image was pruned out from under us → clear, actionable error
        """
        restarted = self._restart_existing(container_id, image_tag)
        if restarted is not None:
            return restarted

        # Fresh-run path — clear any stale forge-<env> container with the same
        # name first, otherwise containers.run raises 409 Conflict.
        self._remove_existing(env_name)
        try:
            if image_tag == "builtin:cli":
                return self.run_cli(env_name)
            if image_tag == "builtin:browser":
                return self.run_browser(env_name)
            cid, port = self.run(env_name, image_tag)
            if port == 0:
                # Defence in depth: a general env should always come back
                # with a real host port. Returning 0 would silently land
                # in the DB as null and break the proxy iframe.
                raise RuntimeError(
                    f"Container started but no host port was bound for {env_name} "
                    f"(image {image_tag!r}). Try rebuilding the environment."
                )
            return cid, port
        except docker.errors.ImageNotFound:
            raise RuntimeError(
                f"Docker image {image_tag!r} is not present locally — the "
                f"environment must be rebuilt before it can be started."
            ) from None

    def _restart_existing(self, container_id: str, image_tag: str) -> tuple[str, int] | None:
        """Start the stopped container and its sidecars, or return None to run fresh."""
        if not container_id:
            return None
        try:
            existing = self._docker.containers.get(container_id)
        except (docker.errors.NotFound, docker.errors.APIError):
            # NullResource and friends: treat the same as NotFound — run fresh.
            return None
        try:
            existing.start()
            existing.reload()
        except docker.errors.APIError:
            # Container exists but won't start (image vanished, etc.). Drop it
            # and fall through to a fresh run.
            self._force_remove(existing)
            return None
        if image_tag == "builtin:cli":
            return existing.id, 0
        # Relay after the browser, since it joins the browser's namespace.
        for sidecar in self._network.sidecars_for(existing):
            sidecar.start()
            sidecar.reload()
        port = self.host_port(existing)
        if port:
            return existing.id, port
        # Port or gateway disappeared (host reboots, or an app started
        # before gateways existed), so run fresh.
        self._force_remove(existing)
        return None

    @staticmethod
    def _force_remove(container) -> None:
        try:
            container.remove(force=True)
        except docker.errors.APIError:
            pass

    def remove(self, container_id: str, image_tag: str | None = None) -> None:
        self.stop(container_id)
        try:
            container = self._docker.containers.get(container_id)
        except docker.errors.NotFound:
            container = None
        if container is not None:
            sidecars = self._network.sidecars_for(container)
            container.remove()
            for sidecar in sidecars:
                sidecar.remove(force=True)
            self._network.remove(container.labels.get("forge.env", ""))
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
            # Gateways, relays, warm spares, and episode forks are helpers,
            # never the environment itself.
            if not env_name or c.labels.get("forge.role"):
                continue
            # list() already inspects each container, so no reload is needed.
            port = self.host_port(c)
            if port:
                result.append((env_name, c.id, port))
        return result

    # ------------------------------------------------------------------
    # Per-episode worlds, delegated to EpisodeSnapshots
    # ------------------------------------------------------------------

    def snapshot_cli(self, env_name: str, container_id: str, run_id: str) -> str:
        return self._snapshots.snapshot_cli(env_name, container_id, run_id)

    def discard_cli_snapshot(self, snapshot: str) -> None:
        self._snapshots.discard_cli_snapshot(snapshot)

    def warm_cli(self, env_name: str, snapshot: str) -> str:
        return self._snapshots.warm_cli(env_name, snapshot)

    @contextmanager
    def cli_episode(self, env_name: str, snapshot: str, *, refill: bool) -> Iterator[str]:
        with self._snapshots.cli_episode(env_name, snapshot, refill=refill) as container_id:
            yield container_id

    def snapshot_initial_state(self, env_name: str, container_id: str, seed: int = 0) -> str:
        return self._snapshots.snapshot_initial_state(env_name, container_id, seed)

    def discard_initial_snapshot(self, snapshot: str) -> None:
        self._snapshots.discard_initial_snapshot(snapshot)

    @contextmanager
    def cloned_episode(self, env_name: str, snapshot: str, **options) -> Iterator[str | tuple[str, int]]:
        with self._snapshots.cloned_episode(env_name, snapshot, **options) as handle:
            yield handle
