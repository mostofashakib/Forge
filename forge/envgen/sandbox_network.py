"""Each environment's private network and the gateways that publish it on loopback.

App and browser containers each sit on their own `internal` network, which
has no route out, so nothing inside can reach the internet or another
environment. Docker cannot publish a port from an internal network, so each
environment gets a gateway: a pinned socat container that publishes the
loopback ports and forwards only to its own environment.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

import docker
import docker.errors

from forge.envgen.container_specs import BROWSER_UI_PORT, FORGE_APP_PORT, gateway_container, loopback_port
from forge.envgen.docker_images import FORGE_GATEWAY_IMAGE

logger = logging.getLogger(__name__)


def sandbox_network_name(env_name: str) -> str:
    return "forge-sandbox-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", env_name)


def container_name(env_name: str) -> str:
    """Return a Docker-safe container name for an environment."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", env_name)
    return f"forge-{safe}"


def gateway_name(env_name: str) -> str:
    return f"{container_name(env_name)}-gw"


def relay_name(env_name: str) -> str:
    return f"{container_name(env_name)}-cdp"


def wait_for_port_binding(container, port_key: str, *, attempts: int = 10, interval: float = 0.3) -> int:
    """Poll until the daemon reports a host-port mapping for `port_key`.

    `containers.run(ports={...})` returns as soon as the container is created,
    but the actual host-port allocation lands a few hundred ms later on a
    busy macOS Docker Desktop. A single `reload()` right after `run()` can
    therefore report `container.ports == {}` even though the binding will
    appear momentarily. Polling instead of a single read avoids storing
    `container_port=None` in the DB, which is the root of the
    'running but iframe shows Container not running' state.

    Returns the host port as an int. Raises RuntimeError with diagnostics
    if no binding appears within `attempts × interval` seconds.
    """
    last_ports: object = "<not yet read>"
    for _ in range(attempts):
        container.reload()
        last_ports = container.ports
        bindings = (last_ports or {}).get(port_key) or []
        if bindings and isinstance(bindings, list):
            entry = bindings[0]
            host_port = entry.get("HostPort") if isinstance(entry, dict) else None
            if host_port:
                try:
                    p = int(host_port)
                    if p > 0:
                        return p
                except (TypeError, ValueError):
                    pass
        time.sleep(interval)
    raise RuntimeError(
        f"Container {container.id[:12]} started but Docker never reported a "
        f"host-port binding for {port_key} after {attempts * interval:.1f}s "
        f"(container.ports={last_ports!r}). The image likely doesn't expose "
        f"that port, or the daemon failed to allocate one — try rebuilding."
    )


class SandboxNetwork:
    """Creates and tears down an environment's network and gateways.

    `client` returns the Docker client lazily, so building a runtime never
    connects to the daemon until a container operation needs it.
    """

    def __init__(self, client: Callable[[], docker.DockerClient]) -> None:
        self._docker = client

    def ensure(self, env_name: str):
        """The environment's internal network, created on first use."""
        name = sandbox_network_name(env_name)
        client = self._docker()
        try:
            return client.networks.get(name)
        except docker.errors.NotFound:
            pass
        try:
            return client.networks.create(
                name, driver="bridge", internal=True,
                labels={"forge.managed": "true", "forge.env": env_name},
            )
        except docker.errors.APIError:
            # Another worker created it between our lookup and create.
            return client.networks.get(name)

    def remove(self, env_name: str) -> None:
        try:
            self._docker().networks.get(sandbox_network_name(env_name)).remove()
        except docker.errors.NotFound:
            pass
        except docker.errors.APIError as exc:
            # Something is still attached. The next run of this env reuses it.
            logger.warning("[container] could not remove network for %s: %s", env_name, exc)

    def start_gateway(
        self,
        env_name: str,
        *,
        target_name: str,
        name: str,
        forwards: dict[int, int],
        network,
        role: str = "gateway",
    ):
        """Publish each port on loopback, forwarding it to `target_name`."""
        script = " & ".join(
            f"socat TCP-LISTEN:{port},fork,reuseaddr TCP:{target_name}:{target_port}"
            for port, target_port in forwards.items()
        ) + " & wait"
        gateway = self._docker().containers.run(**gateway_container(
            env_name,
            image=FORGE_GATEWAY_IMAGE,
            name=name,
            script=script,
            ports={f"{port}/tcp": loopback_port() for port in forwards},
            role=role,
        ))
        network.connect(gateway)
        return gateway

    def start_environment_gateway(self, env_name: str, forwards: dict[int, int], network):
        """The long-lived gateway that forwards only to the environment's own container."""
        return self.start_gateway(
            env_name,
            target_name=container_name(env_name),
            name=gateway_name(env_name),
            forwards=forwards,
            network=network,
        )

    def gateway_for(self, container):
        """The gateway of an app or browser container, or None if it has none."""
        env_name = container.labels.get("forge.env", "")
        try:
            return self._docker().containers.get(gateway_name(env_name))
        except docker.errors.NotFound:
            return None

    def sidecars_for(self, container) -> list:
        """The environment's helper containers that exist: relay, then gateway."""
        env_name = container.labels.get("forge.env", "")
        found = []
        for name in (relay_name(env_name), gateway_name(env_name)):
            try:
                found.append(self._docker().containers.get(name))
            except docker.errors.NotFound:
                pass
        return found

    def host_port(self, container, container_port: int | None = None) -> int | None:
        """The loopback port Forge reaches `container_port` of this environment on.

        App and browser containers publish nothing themselves, so the port
        is their gateway's. It defaults to the app port, or the UI port for a
        browser. CLI containers have no port.
        """
        kind = container.labels.get("forge.type")
        if kind == "cli":
            return None
        if container_port is None:
            container_port = BROWSER_UI_PORT if kind == "browser" else FORGE_APP_PORT
        gateway = self.gateway_for(container)
        if gateway is None:
            return None
        bindings = (gateway.ports or {}).get(f"{container_port}/tcp") or []
        host_port = bindings[0].get("HostPort") if bindings else None
        return int(host_port) if host_port else None
