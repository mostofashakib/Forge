"""Ports, resource limits and launch settings for each kind of environment container.

Each `*_container` function returns the keyword arguments for
`docker.containers.run`, so the long-lived environment and its per-episode
clones share one definition of isolation and limits.
"""
from __future__ import annotations

import os

from forge.runtime.context import SimClock

# Forge always publishes 8000/tcp from the container, so the app inside
# must listen on the same port — otherwise the host-port binding routes
# to nothing and the iframe shows "Container not running".
FORGE_APP_PORT = 8000
# The browser image serves its KasmVNC UI and Chromium's DevTools here.
# Chromium binds DevTools to its own loopback whatever flags it gets, so a
# relay inside the browser's network namespace re-serves it on the relay port.
BROWSER_UI_PORT = 3000
BROWSER_CDP_PORT = 9222
BROWSER_CDP_RELAY_PORT = 9223

# Runtime limits are intentionally conservative defaults for generated code.
# They can be overridden for larger local experiments without changing code.
DEFAULT_CONTAINER_MEMORY = "1g"
DEFAULT_BROWSER_MEMORY = "2g"
DEFAULT_CLI_MEMORY = "1g"
DEFAULT_CONTAINER_NANO_CPUS = 1_000_000_000
DEFAULT_CONTAINER_PIDS = 256
GENERAL_MEMORY_LIMIT = os.environ.get("FORGE_CONTAINER_MEMORY", DEFAULT_CONTAINER_MEMORY)
BROWSER_MEMORY_LIMIT = os.environ.get("FORGE_BROWSER_MEMORY", DEFAULT_BROWSER_MEMORY)
CLI_MEMORY_LIMIT = os.environ.get("FORGE_CLI_MEMORY", DEFAULT_CLI_MEMORY)
GATEWAY_MEMORY_LIMIT = "64m"
CPU_LIMIT = int(os.environ.get("FORGE_CONTAINER_NANO_CPUS", str(DEFAULT_CONTAINER_NANO_CPUS)))
PID_LIMIT = int(os.environ.get("FORGE_CONTAINER_PIDS", str(DEFAULT_CONTAINER_PIDS)))

# Python salts str hashes per process, so set iteration order changes on every
# container start. A fixed seed makes it depend only on what was inserted.
PYTHON_HASH_SEED = "0"

# Every process in the CLI container starts its clock at the SimClock epoch.
CLI_FAKETIME_ENV = {
    "LD_PRELOAD": "/usr/local/lib/libfaketime.so.1",
    "FAKETIME": SimClock().now().strftime("@%Y-%m-%d %H:%M:%S"),
}

HARDENED = {"cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"]}

_BROWSER_CHROME_FLAGS = "--remote-debugging-port=9222 --remote-debugging-address=0.0.0.0 --no-sandbox"


def loopback_port() -> tuple[str, None]:
    """Ask Docker for a random host port bound only to loopback."""
    return ("127.0.0.1", None)


def _determinism() -> str:
    return os.environ.get("FORGE_DETERMINISM", "on")


def app_container(
    env_name: str, *, image: str, name: str, network: str, redis_url: str, labels: dict, restart_policy: dict,
) -> dict:
    """A generated or premade HTTP app: no published port and no route out."""
    return {
        "image": image,
        "name": name,
        "detach": True,
        "network": network,
        "environment": {
            "TZ": "UTC",
            "PYTHONHASHSEED": PYTHON_HASH_SEED,
            "REDIS_URL": redis_url,
            "FORGE_ENV_NAME": env_name,
            "FORGE_DETERMINISM": _determinism(),
        },
        "labels": {"forge.env": env_name, "forge.managed": "true", **labels},
        "restart_policy": restart_policy,
        "mem_limit": GENERAL_MEMORY_LIMIT,
        "nano_cpus": CPU_LIMIT,
        "pids_limit": PID_LIMIT,
        **HARDENED,
        "init": True,
    }


def browser_container(
    env_name: str, *, image: str, name: str, network: str, labels: dict, restart_policy: dict, **extra,
) -> dict:
    """Chromium with KasmVNC. DevTools is exposed so agents drive it over CDP."""
    return {
        "image": image,
        "name": name,
        "detach": True,
        "network": network,
        "environment": {
            "PUID": "1000",
            "PGID": "1000",
            "TZ": "UTC",
            "FORGE_DETERMINISM": _determinism(),
            "CHROME_CLI": _BROWSER_CHROME_FLAGS,
        },
        "shm_size": "1g",
        "labels": {"forge.env": env_name, "forge.managed": "true", "forge.type": "browser", **labels},
        "restart_policy": restart_policy,
        "mem_limit": BROWSER_MEMORY_LIMIT,
        "nano_cpus": CPU_LIMIT,
        "pids_limit": PID_LIMIT,
        **extra,
    }


def cli_container(env_name: str, *, image: str, name: str, labels: dict) -> dict:
    """One CLI shell. The environment, its warm spare, and every episode fork share this."""
    return {
        "image": image,
        "name": name,
        "command": ["tail", "-f", "/dev/null"],
        "detach": True,
        "labels": {"forge.env": env_name, "forge.managed": "true", "forge.type": "cli", **labels},
        "restart_policy": {"Name": "unless-stopped"},
        "mem_limit": CLI_MEMORY_LIMIT,
        "nano_cpus": CPU_LIMIT,
        "pids_limit": PID_LIMIT,
        "init": True,
        # The agent reaches the shell through `docker exec`, so the
        # container needs no network. Without one, commands see a fixed
        # world instead of whatever the internet serves today.
        "network_mode": "none",
        "environment": {
            "TZ": "UTC",
            "PYTHONHASHSEED": PYTHON_HASH_SEED,
            **CLI_FAKETIME_ENV,
            "FORGE_DETERMINISM": _determinism(),
        },
    }


def gateway_container(
    env_name: str, *, image: str, name: str, script: str, ports: dict, role: str,
) -> dict:
    """A socat sidecar that publishes loopback ports and forwards them inward."""
    return {
        "image": image,
        "name": name,
        "entrypoint": ["/bin/sh", "-c"],
        "command": [script],
        "detach": True,
        "ports": ports,
        "labels": {"forge.env": env_name, "forge.managed": "true", "forge.role": role},
        "restart_policy": {"Name": "unless-stopped" if role == "gateway" else "no"},
        "mem_limit": GATEWAY_MEMORY_LIMIT,
        "nano_cpus": CPU_LIMIT,
        "pids_limit": PID_LIMIT,
        **HARDENED,
    }
