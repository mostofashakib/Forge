"""Give every episode its own world, forked from an immutable snapshot image.

CLI runs freeze the shell once per run and fork each episode from it, with a
warm spare started ahead of time. App and browser episodes clone the seeded
initial snapshot behind their own short-lived gateway.
"""
from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import docker
import docker.errors

from forge.envgen.container_specs import (
    BROWSER_CDP_PORT,
    BROWSER_CDP_RELAY_PORT,
    BROWSER_UI_PORT,
    FORGE_APP_PORT,
    HARDENED,
    app_container,
    browser_container,
    cli_container,
)
from forge.envgen.docker_cli import DockerCLI
from forge.envgen.docker_images import cli_snapshot_tag, initial_snapshot_tag
from forge.envgen.sandbox_network import SandboxNetwork, container_name, wait_for_port_binding


class EpisodeSnapshots:
    def __init__(
        self,
        client: Callable[[], docker.DockerClient],
        network: SandboxNetwork,
        cli: DockerCLI,
        redis_url: str,
    ) -> None:
        self._docker = client
        self._network = network
        self._cli = cli
        self._redis_url = redis_url

    # ------------------------------------------------------------------
    # CLI: one snapshot per run, one fork per episode
    # ------------------------------------------------------------------

    def run_cli_container(self, env_name: str, name: str, image: str, **labels: str):
        """Start one CLI shell with the isolation every CLI container shares."""
        return self._docker().containers.run(**cli_container(env_name, image=image, name=name, labels=labels))

    def snapshot_cli(self, env_name: str, container_id: str, run_id: str) -> str:
        """Freeze the environment's shell, including any terminal setup, for a run."""
        tag = cli_snapshot_tag(env_name, run_id)
        self._cli.run("commit", container_id, tag)
        return tag

    def discard_cli_snapshot(self, snapshot: str) -> None:
        """Drop a finished run's snapshot image. A missing image is fine."""
        self._cli.discard("rmi", "-f", snapshot)

    def warm_cli(self, env_name: str, snapshot: str) -> str:
        """Start the spare the next episode takes, so it never waits for a boot."""
        name = f"{container_name(env_name)}-warm"
        self._remove_if_present(name)
        container = self.run_cli_container(
            env_name, name, snapshot, **{"forge.role": "warm", "forge.snapshot": snapshot}
        )
        return container.id

    @contextmanager
    def cli_episode(self, env_name: str, snapshot: str, *, refill: bool) -> Iterator[str]:
        """Yield a fresh shell forked from the run's snapshot, then discard it.

        The warm spare is taken when it was forked from this snapshot. After
        the episode, `refill` starts the next spare off the critical path.
        """
        base = container_name(env_name)
        episode_name = f"{base}-episode"
        self._remove_if_present(episode_name)
        container = self._take_warm_spare(f"{base}-warm", snapshot, episode_name)
        if container is None:
            container = self.run_cli_container(
                env_name, episode_name, snapshot, **{"forge.role": "episode"}
            )
        try:
            yield container.id
        finally:
            container.remove(force=True)
            if refill:
                self.warm_cli(env_name, snapshot)

    def _take_warm_spare(self, warm_name: str, snapshot: str, episode_name: str):
        """Rename a running spare forked from `snapshot` into the episode; drop any other spare."""
        try:
            warm = self._docker().containers.get(warm_name)
        except docker.errors.NotFound:
            return None
        if warm.labels.get("forge.snapshot") == snapshot and warm.status == "running":
            warm.rename(episode_name)
            return warm
        warm.remove(force=True)
        return None

    # ------------------------------------------------------------------
    # App and browser: clone the seeded initial snapshot per episode
    # ------------------------------------------------------------------

    def snapshot_initial_state(self, env_name: str, container_id: str, seed: int = 0) -> str:
        """Freeze the environment container with seed data into an immutable snapshot image."""
        tag = initial_snapshot_tag(env_name, seed)
        self._cli.run("commit", container_id, tag)
        return tag

    def discard_initial_snapshot(self, snapshot: str) -> None:
        """Drop an initial state snapshot image. A missing image is fine."""
        self._cli.discard("rmi", "-f", snapshot)

    @contextmanager
    def cloned_episode(
        self,
        env_name: str,
        snapshot: str,
        *,
        env_type: str = "general",
        episode_id: str | None = None,
        refill: bool = False,
    ) -> Iterator[str | tuple[str, int]]:
        """Yield an isolated container cloned from an immutable snapshot, then discard it.

        For CLI environments, yields container_id.
        For general (HTTP) and browser environments, yields (container_id, host_port).
        """
        if env_type == "cli":
            with self.cli_episode(env_name, snapshot, refill=refill) as cid:
                yield cid
            return

        ep_token = re.sub(r"[^a-zA-Z0-9_.-]", "-", episode_id or secrets.token_hex(4))
        ep_name = f"{container_name(env_name)}-ep-{ep_token}"
        network = self._network.ensure(env_name)
        self._remove_if_present(ep_name)

        labels = {"forge.role": "episode"}
        restart = {"Name": "no"}
        if env_type == "browser":
            spec = browser_container(
                env_name, image=snapshot, name=ep_name, network=network.name,
                labels=labels, restart_policy=restart, **HARDENED, init=True,
            )
            forwards = {BROWSER_UI_PORT: BROWSER_UI_PORT, BROWSER_CDP_PORT: BROWSER_CDP_RELAY_PORT}
            published = BROWSER_UI_PORT
        else:  # general / premade HTTP
            spec = app_container(
                env_name, image=snapshot, name=ep_name, network=network.name, redis_url=self._redis_url,
                labels={"forge.type": "general", **labels}, restart_policy=restart,
            )
            forwards = {FORGE_APP_PORT: FORGE_APP_PORT}
            published = FORGE_APP_PORT

        container = self._docker().containers.run(**spec)
        gateway = self._network.start_gateway(
            env_name, target_name=ep_name, name=f"{ep_name}-gw",
            forwards=forwards, network=network, role="episode-gateway",
        )
        try:
            port = wait_for_port_binding(gateway, f"{published}/tcp", attempts=10, interval=0.3)
            yield container.id, port
        finally:
            for leftover in (container, gateway):
                try:
                    leftover.remove(force=True)
                except Exception:
                    pass

    def _remove_if_present(self, name: str) -> None:
        try:
            self._docker().containers.get(name).remove(force=True)
        except docker.errors.NotFound:
            pass
