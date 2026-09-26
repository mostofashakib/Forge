from __future__ import annotations
import hashlib
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import docker
import docker.errors

from forge.logging_utils import redact_sensitive_text

from forge.envgen.runtime_lock import RUNTIME_LOCK
from forge.runtime.context import SimClock
from forge.runtime.network_isolation import check_generated_env
from forge.settings import redis_url


# Single canonical base image for every generated env. The LLM-generated
# Dockerfile gets its FROM line normalised to this, so all builds depend
# on exactly one image. We pre-warm it at worker startup, which means the
# user-triggered build path always finds it cached and never contacts
# Docker Hub — making the system immune to transient Hub EOF outages.
#
# Every image is pinned by digest. A tag moves when upstream republishes it,
# so the same environment would otherwise build on different bytes each week.
FORGE_PYTHON_BASE = os.environ.get(
    "FORGE_PYTHON_BASE_IMAGE",
    "python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f",
)
FORGE_CLI_IMAGE = os.environ.get(
    "FORGE_CLI_IMAGE",
    "ubuntu:22.04@sha256:b8b6ee6aa931ecd9d0d952abc34dc0e5f7c6a30c6bb71b079fe399fde0329c02",
)
FORGE_BROWSER_IMAGE = os.environ.get(
    "FORGE_BROWSER_IMAGE",
    "lscr.io/linuxserver/chromium:latest"
    "@sha256:cf6200ccdcb224feaf5d3bde4ce45b3783c926a7496c98059cae1e0db78e5b2f",
)

# App and browser containers each sit on their own `internal` network, which
# has no route out, so nothing inside can reach the internet or another
# environment. Docker cannot publish a port from an internal network, so each
# environment gets a gateway: a pinned socat container that publishes the
# loopback ports and forwards only to its own environment.
def sandbox_network_name(env_name: str) -> str:
    return "forge-sandbox-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", env_name)


FORGE_GATEWAY_IMAGE = os.environ.get(
    "FORGE_GATEWAY_IMAGE",
    "alpine/socat:1.8.0.3@sha256:beb4a68d9e4fe6b0f21ea774a0fde6c31f580dde6368939ed70100c5385b015e",
)
_GATEWAY_MEMORY_LIMIT = "64m"

# CLI grading uses only this toolbox: a static BusyBox from a pinned image,
# copied into a volume and mounted read-only into the grader. Static binaries
# ignore LD_PRELOAD and /etc/ld.so.preload, so nothing the agent wrote in its
# container can change what an assertion's `grep` or `test` reports.
FORGE_GRADER_TOOLBOX_IMAGE = os.environ.get(
    "FORGE_GRADER_TOOLBOX_IMAGE",
    "busybox:1.37.0-musl@sha256:5cec3fc171c87218698e85a52af7087de727372aae264a787b8112901a5b0092",
)
_GRADER_TOOLBOX_DIR = "/opt/forge-grader"
_GRADER_PATH = (
    f"{_GRADER_TOOLBOX_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

# Standard base images Forge needs locally. Pre-warmed at Celery worker
# startup so user-triggered builds never wait on Hub.
STANDARD_BASE_IMAGES: tuple[str, ...] = (
    FORGE_PYTHON_BASE,
    FORGE_CLI_IMAGE,
    FORGE_BROWSER_IMAGE,
    FORGE_GATEWAY_IMAGE,
    FORGE_GRADER_TOOLBOX_IMAGE,
)

# The CLI environment runs a prebuilt image: the pinned Ubuntu base plus
# libfaketime. Its tag is derived from the Dockerfile, so editing the file
# builds a new image instead of reusing a stale one.
CLI_DOCKERFILE = Path(__file__).with_name("cli_image") / "Dockerfile"


def _cli_image_tag(dockerfile_text: str) -> str:
    digest = hashlib.sha256(f"{FORGE_CLI_IMAGE}\n{dockerfile_text}".encode()).hexdigest()
    return f"forge-cli:{digest[:16]}"


CLI_RUNTIME_IMAGE = _cli_image_tag(CLI_DOCKERFILE.read_text())

def cli_snapshot_tag(env_name: str, run_id: str) -> str:
    """The image a run's CLI episodes fork from: the environment, frozen at run start."""
    repo = re.sub(r"[^a-z0-9._-]", "-", env_name.lower())
    tag = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)[:128]
    return f"forge-cli-snapshot-{repo}:{tag}"


# Every process in the CLI container starts its clock at the SimClock epoch.
_CLI_FAKETIME_ENV = {
    "LD_PRELOAD": "/usr/local/lib/libfaketime.so.1",
    "FAKETIME": SimClock().now().strftime("@%Y-%m-%d %H:%M:%S"),
}

# The only install step a Forge Dockerfile may contain. `--require-hashes`
# makes pip refuse anything the lock does not pin to an exact artifact.
_LOCKED_INSTALL = "RUN pip install --no-cache-dir --require-hashes -r requirements.txt"

_DOCKERFILE = f"""\
FROM {FORGE_PYTHON_BASE}
WORKDIR /app
COPY requirements.txt .
{_LOCKED_INSTALL}
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""


# Matches FROM lines whose base is a python:* tag (any tag, with or without digest),
# optionally followed by `AS <stage>`. Non-python bases are left untouched.
_PYTHON_FROM_RE = re.compile(
    r"^(\s*FROM\s+)python:[^\s]+(\s+AS\s+\S+)?\s*$",
    re.IGNORECASE,
)


# Forge always publishes 8000/tcp from the container, so the app inside
# must listen on the same port — otherwise the host-port binding routes
# to nothing and the iframe shows "Container not running".
FORGE_APP_PORT = 8000
# The browser image serves its KasmVNC UI and Chromium's DevTools here.
# Chromium binds DevTools to its own loopback whatever flags it gets, so a
# relay inside the browser's network namespace re-serves it on the relay port.
_BROWSER_UI_PORT = 3000
_BROWSER_CDP_PORT = 9222
_BROWSER_CDP_RELAY_PORT = 9223

# Runtime limits are intentionally conservative defaults for generated code.
# They can be overridden for larger local experiments without changing code.
_GENERAL_MEMORY_LIMIT = os.environ.get("FORGE_CONTAINER_MEMORY", "1g")
_BROWSER_MEMORY_LIMIT = os.environ.get("FORGE_BROWSER_MEMORY", "2g")
_CLI_MEMORY_LIMIT = os.environ.get("FORGE_CLI_MEMORY", "1g")
_CPU_LIMIT = int(os.environ.get("FORGE_CONTAINER_NANO_CPUS", "1000000000"))
_PID_LIMIT = int(os.environ.get("FORGE_CONTAINER_PIDS", "256"))


# Python salts str hashes per process, so set iteration order changes on every
# container start. A fixed seed makes it depend only on what was inserted.
_PYTHON_HASH_SEED = "0"


def _loopback_port() -> tuple[str, None]:
    """Ask Docker for a random host port bound only to loopback."""
    return ("127.0.0.1", None)

def _write_locked_requirements(app_dir: Path) -> bool:
    """Make requirements.txt the runtime lock, whatever the LLM listed.

    An unpinned requirements file resolves against whatever PyPI serves on
    build day. The correctness gate rejects generated code that imports a
    package the lock lacks, so replacing the file never drops a real need.
    Returns True if the file was created or changed.
    """
    lock = RUNTIME_LOCK.read_text()
    req_file = app_dir / "requirements.txt"
    if req_file.exists() and req_file.read_text() == lock:
        return False
    req_file.write_text(lock)
    return True


# `pip install` in any spelling (pip, pip3, python -m pip).
_PIP_INSTALL_RE = re.compile(r"\bpip3?\s+install\b")
# System package managers resolve against live distro mirrors.
_SYSTEM_INSTALL_RE = re.compile(r"\b(apt-get|apt|apk)\s+(update|install|add)\b")


def _dockerfile_instructions(text: str) -> list[str]:
    """Split a Dockerfile into instructions, keeping `\\` continuations whole."""
    instructions: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        current.append(line)
        if not line.rstrip().endswith("\\"):
            instructions.append("\n".join(current))
            current = []
    if current:
        instructions.append("\n".join(current))
    return instructions


def _normalise_dockerfile_install(dockerfile: Path) -> bool:
    """Reduce every install step to the single hashed install of the lock.

    The first RUN that calls pip becomes `_LOCKED_INSTALL`. Later pip RUNs
    and every system package install are dropped: the lock is the whole
    dependency set, and nothing in it needs a system package.
    Returns True if the file was modified.
    """
    original = dockerfile.read_text()
    kept: list[str] = []
    installed = False
    for instruction in _dockerfile_instructions(original):
        is_run = instruction.lstrip().upper().startswith("RUN ")
        if is_run and _PIP_INSTALL_RE.search(instruction):
            if not installed:
                kept.append(_LOCKED_INSTALL)
                installed = True
            continue
        if is_run and _SYSTEM_INSTALL_RE.search(instruction):
            continue
        kept.append(instruction)
    text = "\n".join(kept) + "\n"
    if text == original:
        return False
    dockerfile.write_text(text)
    return True

# Match a whole EXPOSE line (case-insensitive). Uses `[ \t]` instead of `\s`
# for the trailing whitespace so the match can't slurp across the newline
# into the next directive.
_EXPOSE_RE = re.compile(r"^[ \t]*EXPOSE[ \t]+\d+[ \t]*$", re.MULTILINE | re.IGNORECASE)

# Match `--port` followed (after any non-word run — spaces, commas, equals,
# quotes) by a port number. Catches every shell, JSON-array, and
# equals-form CMD the LLM might emit:
#   CMD uvicorn --port 5000
#   CMD ["uvicorn", "--port", "5000"]
#   CMD uvicorn --port=5000
_PORT_FLAG_RE = re.compile(r"(--port)(\W+?)(\d+)")


def _normalise_dockerfile_port(dockerfile: Path) -> bool:
    """Force every EXPOSE and `--port` in the Dockerfile to FORGE_APP_PORT.

    The LLM that writes the Dockerfile picks ports probabilistically — sometimes
    8000, sometimes 5000, sometimes 8080. Forge always publishes 8000/tcp on the
    container side, so anything else means a dead port mapping. We rewrite the
    file to keep the two ends consistent regardless of what the LLM chose.

    Returns True if any change was made.
    """
    original = dockerfile.read_text()
    text = original

    # Rewrite any EXPOSE line to use the canonical port.
    text, expose_subs = _EXPOSE_RE.subn(f"EXPOSE {FORGE_APP_PORT}", text)
    # Add EXPOSE if missing entirely.
    if expose_subs == 0:
        text = text.rstrip() + f"\nEXPOSE {FORGE_APP_PORT}\n"

    # Rewrite any `--port N` (in any quoting / spacing form) to the canonical port.
    text = _PORT_FLAG_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{FORGE_APP_PORT}",
        text,
    )

    if text != original:
        dockerfile.write_text(text)
        return True
    return False


def _normalise_dockerfile_base(dockerfile: Path) -> bool:
    """Rewrite any `FROM python:*` line to `FROM <FORGE_PYTHON_BASE>`.

    Returns True if the file was modified. The LLM that generates the
    Dockerfile picks an arbitrary python tag (e.g. 3.11-slim, 3.12-bookworm),
    each requiring its own Hub pull. Normalising to a single canonical
    base means we only need to keep ONE image warm locally.
    """
    original = dockerfile.read_text()
    new_lines: list[str] = []
    changed = False
    for line in original.splitlines(keepends=True):
        m = _PYTHON_FROM_RE.match(line.rstrip("\r\n"))
        if m:
            stage_suffix = m.group(2) or ""
            ending = "\n" if line.endswith("\n") else ""
            replacement = f"{m.group(1)}{FORGE_PYTHON_BASE}{stage_suffix}{ending}"
            if replacement != line:
                changed = True
                new_lines.append(replacement)
                continue
        new_lines.append(line)
    if changed:
        dockerfile.write_text("".join(new_lines))
    return changed


def _parse_from_image(dockerfile: Path) -> str | None:
    """Return the base image name from the first FROM line in a Dockerfile."""
    for line in dockerfile.read_text().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            parts = stripped.split()
            if len(parts) >= 2:
                image = parts[1]
                # Strip build-stage alias (e.g. "python:3.12-slim AS builder")
                return image if image.lower() != "scratch" else None
    return None


_PERMANENT_PULL_ERRORS = (
    "not found",
    "unauthorized",
    "pull access denied",
    "does not exist",
    "invalid reference",
    "no such image",
)


# Docker Hub mirrors that serve the official `library/*` images on
# independent infrastructure. When registry-1.docker.io is throwing EOFs,
# these typically still respond. We fall back through them in order and
# `docker tag` the successful pull as the canonical name so callers stay
# blind to which registry actually served the image.
_HUB_MIRRORS: tuple[str, ...] = (
    "public.ecr.aws/docker",  # AWS Public ECR mirror of Docker Hub official images
    "mirror.gcr.io",          # Google's pull-through cache of Docker Hub
)


def _wait_for_port_binding(container, port_key: str, *, attempts: int = 10, interval: float = 0.3) -> int:
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


def _is_hub_image(image: str) -> bool:
    """Return True when `image` resolves to docker.io (no explicit registry).

    Docker treats the first slash-separated component as a registry only if
    it contains a '.' or ':' (or is 'localhost'). A bare 'repo:tag' with no
    slash is always a Hub library reference.
    """
    if "/" not in image:
        return True
    head = image.split("/", 1)[0]
    return ("." not in head) and (":" not in head) and (head != "localhost")


def _mirror_ref_for(image: str, mirror_prefix: str) -> str:
    """Rewrite a Hub image to its equivalent on the given mirror.

    'python:3.12-slim'      → '<mirror>/library/python:3.12-slim'
    'user/repo:tag'         → '<mirror>/user/repo:tag'
    """
    if "/" in image:
        # user/repo:tag → mirror/user/repo:tag
        return f"{mirror_prefix}/{image}"
    # bare repo:tag → mirror/library/repo:tag
    return f"{mirror_prefix}/library/{image}"


def _docker_tag(source: str, target: str) -> None:
    """Add an alias tag in the local cache so callers can reference the canonical name."""
    subprocess.run(
        ["docker", "tag", source, target],
        check=True, capture_output=True, text=True,
    )


def _image_cached_locally(image: str) -> bool:
    """Return True if the image is already in the local Docker daemon cache."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True, check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


# Generous enough for a cold pip install, but a hung build must not hold a
# worker slot forever.
_BUILD_TIMEOUT_S = 900


def _pull_with_retry(image: str, max_attempts: int = 5, pull_timeout: int = 120) -> None:
    """Pull a Docker image via CLI, retrying on transient network errors.

    Uses exponential backoff capped at 30 s (1 s, 2 s, 4 s, 8 s, 16 s…).
    Each individual pull is capped at `pull_timeout` seconds (default 120 s)
    so a hung Docker daemon connection never blocks the worker indefinitely.
    Surfaces Docker's stderr so the caller can see the actual error, and
    skips retries immediately when the error is permanent (image not found,
    auth denied).

    NOTE: this is a fallback. Hub-flakiness is primarily handled by
    pre-warming the standard base images at Celery worker boot
    (see `prewarm_standard_base_images`), so the user-triggered build
    path always hits the local cache.
    """
    last_exc: Exception | None = None
    last_output: str = ""
    for attempt in range(max_attempts):
        try:
            subprocess.run(
                ["docker", "pull", image],
                check=True,
                capture_output=True,
                text=True,
                timeout=pull_timeout,
            )
            return
        except subprocess.TimeoutExpired as exc:
            last_exc = exc
            last_output = f"pull timed out after {pull_timeout}s"
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            last_output = redact_sensitive_text(
                ((exc.stderr or "") + (exc.stdout or "")).strip()
            )
            if any(p in last_output.lower() for p in _PERMANENT_PULL_ERRORS):
                break  # retrying won't help
        if attempt < max_attempts - 1:
            delay = min(2 ** attempt, 30)  # 1 s, 2 s, 4 s, 8 s, 16 s … capped at 30 s
            time.sleep(delay)
    docker_msg = f"\nDocker output: {last_output}" if last_output else ""
    raise RuntimeError(
        f"Failed to pull {image} after {max_attempts} attempts: {last_exc}{docker_msg}"
    ) from last_exc


def pull_image(image: str) -> None:
    """Pull a Docker image with full Hub-mirror and transport fallback.

    For Docker Hub references, the order is:
      1. `docker pull` against docker.io (canonical, fastest path).
      2. `docker pull` against public.ecr.aws/docker (AWS mirror of Hub).
      3. `docker pull` against mirror.gcr.io (Google mirror of Hub).
      4. **Direct HTTPS via httpx** (`pull_via_http`) — bypasses dockerd's
         transport entirely. This catches the case where the daemon's HTTP/2
         client is unstable (MTU mismatch, broken IPv6, idle-stream resets),
         which presents as `EOF` errors against multiple unrelated registries
         simultaneously. httpx forces HTTP/1.1 over a fresh TLS stack, so it
         succeeds where dockerd's pull pipeline keeps tearing connections.

    On success of a mirror, `docker tag` aliases it under the canonical name.
    On success of the HTTPS path, `docker load` already imports it under the
    canonical name. Callers stay blind to which path served the image.

    For non-Hub references (explicit registry like `lscr.io/...`), only
    the canonical `docker pull` is tried — there's no Hub mirror or HTTPS
    fallback for those.
    """
    import logging
    log = logging.getLogger(__name__)

    # Already cached locally — nothing to do.
    if _image_cached_locally(image):
        return

    if not _is_hub_image(image):
        _pull_with_retry(image)
        return

    errors: list[str] = []

    # 1) Canonical docker.io
    try:
        _pull_with_retry(image)
        return
    except RuntimeError as exc:
        errors.append(f"docker.io → {redact_sensitive_text(exc)}")
        # A registry miss is expected fallback behavior, not an operational
        # warning. Only surface a warning if every pull transport fails.
        log.info("[pull] docker.io unavailable for %s; trying mirrors", image)

    # 2) Hub mirrors (AWS ECR, Google GCR mirror)
    for mirror in _HUB_MIRRORS:
        mirror_ref = _mirror_ref_for(image, mirror)
        try:
            _pull_with_retry(mirror_ref)
            # `docker tag` refuses a digest target. The name alone is enough:
            # the digest ref then resolves to this content-addressed image.
            _docker_tag(mirror_ref, image.split("@", 1)[0])
            log.info("[pull] %s served by %s", image, mirror)
            return
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            errors.append(f"{mirror} → {redact_sensitive_text(exc)}")
            log.info("[pull] %s unavailable for %s", mirror, image)

    # The HTTPS loader rebuilds the manifest locally, so its digest can never
    # match a pinned one. Serving it would silently unpin the build.
    if "@" in image:
        raise RuntimeError(
            f"Failed to pull {image} from docker.io or any mirror. Direct HTTPS "
            "is skipped because it cannot preserve a pinned digest:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    # 3) Direct HTTPS via httpx — independent transport, independent of
    #    whatever's wrong with the Docker daemon's HTTP/2 client.
    try:
        from forge.envgen._image_pull_http import pull_via_http
        log.info(
            "[pull] all docker-pull paths failed for %s; falling back to direct HTTPS",
            image,
        )
        pull_via_http(image)
        log.info("[pull] %s served by direct HTTPS (bypassed dockerd)", image)
        return
    except Exception as exc:  # noqa: BLE001 — last-resort fallback, surface whatever broke
        safe_error = redact_sensitive_text(exc)
        errors.append(f"direct-https → {safe_error}")
        log.warning("[pull] direct HTTPS also failed for %s: %s", image, safe_error)

    raise RuntimeError(
        f"Failed to pull {image} from docker.io, any mirror, or direct HTTPS:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


class ContainerRuntime:
    def __init__(self, docker_client: docker.DockerClient | None = None) -> None:
        self._docker_client = docker_client
        self._redis_url = redis_url()

    @staticmethod
    def _container_name(env_name: str) -> str:
        """Return a Docker-safe container name for an environment."""
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", env_name)
        return f"forge-{safe}"

    @classmethod
    def _gateway_name(cls, env_name: str) -> str:
        return f"{cls._container_name(env_name)}-gw"

    @classmethod
    def _relay_name(cls, env_name: str) -> str:
        return f"{cls._container_name(env_name)}-cdp"

    @property
    def _docker(self) -> docker.DockerClient:
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    def build(self, env_name: str, app_dir: Path) -> str:
        dockerfile = app_dir / "Dockerfile"
        if not dockerfile.exists():
            dockerfile.write_text(_DOCKERFILE)
        else:
            # Normalise any LLM-generated `FROM python:X` to the single
            # canonical Forge base. This guarantees every build depends
            # on the one image we pre-warm, so Hub is never on the hot path.
            _normalise_dockerfile_base(dockerfile)
            # Force EXPOSE / --port to FORGE_APP_PORT so the in-container
            # listener and the host-side port mapping never disagree, no
            # matter what port the LLM picked.
            _normalise_dockerfile_port(dockerfile)
            # Only the hashed lock may be installed, so nothing is resolved
            # at build time.
            _normalise_dockerfile_install(dockerfile)

        # The lock replaces whatever the LLM listed. It also covers the
        # runtime deps generated code needs, which the LLM that writes
        # requirements.txt routinely forgets.
        _write_locked_requirements(app_dir)

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
        base_image = _parse_from_image(dockerfile)
        if base_image:
            pull_image(base_image)

        safe_name = env_name.replace("_", "-").lower()
        tag = f"forge-env-{safe_name}:latest"
        # Use CLI directly — the Python SDK credential-helper resolution fails
        # when optional helpers (e.g. docker-credential-gcloud) are configured
        # but not authenticated, even for Docker Hub images.
        try:
            subprocess.run(
                ["docker", "build", "-t", tag, "--rm", str(app_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=_BUILD_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"docker build timed out after {_BUILD_TIMEOUT_S}s") from exc
        except subprocess.CalledProcessError as exc:
            output = (exc.stderr or exc.stdout or "(no output)").strip()
            raise RuntimeError(
                f"docker build failed (exit {exc.returncode}):\n{output}"
            ) from exc
        return tag

    def _remove_existing(self, env_name: str) -> None:
        """Remove every container that belongs to an environment, if present."""
        base = self._container_name(env_name)
        for name in (
            self._relay_name(env_name), base, self._gateway_name(env_name),
            f"{base}-warm", f"{base}-episode",
        ):
            try:
                self._docker.containers.get(name).remove(force=True)
            except docker.errors.NotFound:
                pass

    def _sandbox_network(self, env_name: str):
        """The environment's internal network, created on first use."""
        name = sandbox_network_name(env_name)
        try:
            return self._docker.networks.get(name)
        except docker.errors.NotFound:
            pass
        try:
            return self._docker.networks.create(
                name, driver="bridge", internal=True,
                labels={"forge.managed": "true", "forge.env": env_name},
            )
        except docker.errors.APIError:
            # Another worker created it between our lookup and create.
            return self._docker.networks.get(name)

    def _start_gateway(self, env_name: str, forwards: dict[int, int], network):
        """Publish each port on loopback, forwarding it to a port on the environment only."""
        target = self._container_name(env_name)
        ports = tuple(forwards)
        script = " & ".join(
            f"socat TCP-LISTEN:{port},fork,reuseaddr TCP:{target}:{target_port}"
            for port, target_port in forwards.items()
        ) + " & wait"
        gateway = self._docker.containers.run(
            image=FORGE_GATEWAY_IMAGE,
            name=self._gateway_name(env_name),
            entrypoint=["/bin/sh", "-c"],
            command=[script],
            detach=True,
            ports={f"{port}/tcp": _loopback_port() for port in ports},
            labels={"forge.env": env_name, "forge.managed": "true", "forge.role": "gateway"},
            restart_policy={"Name": "unless-stopped"},
            mem_limit=_GATEWAY_MEMORY_LIMIT,
            nano_cpus=_CPU_LIMIT,
            pids_limit=_PID_LIMIT,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
        )
        network.connect(gateway)
        return gateway

    def _remove_network(self, env_name: str) -> None:
        try:
            self._docker.networks.get(sandbox_network_name(env_name)).remove()
        except docker.errors.NotFound:
            pass
        except docker.errors.APIError as exc:
            # Something is still attached. The next run of this env reuses it.
            import logging
            logging.getLogger(__name__).warning(
                "[container] could not remove network for %s: %s", env_name, exc
            )

    def _gateway_for(self, container):
        """The gateway of an app or browser container, or None if it has none."""
        env_name = container.labels.get("forge.env", "")
        try:
            return self._docker.containers.get(self._gateway_name(env_name))
        except docker.errors.NotFound:
            return None

    def _sidecars_for(self, container) -> list:
        """The environment's helper containers that exist: relay, then gateway."""
        env_name = container.labels.get("forge.env", "")
        found = []
        for name in (self._relay_name(env_name), self._gateway_name(env_name)):
            try:
                found.append(self._docker.containers.get(name))
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
            container_port = _BROWSER_UI_PORT if kind == "browser" else FORGE_APP_PORT
        gateway = self._gateway_for(container)
        if gateway is None:
            return None
        bindings = (gateway.ports or {}).get(f"{container_port}/tcp") or []
        host_port = bindings[0].get("HostPort") if bindings else None
        return int(host_port) if host_port else None

    def run_cli(self, env_name: str) -> tuple[str, int]:
        """Spin up an Ubuntu 22.04 shell container (no HTTP port)."""
        image = ensure_cli_image()
        self._remove_existing(env_name)
        container = self._run_cli_container(env_name, self._container_name(env_name), image)
        return container.id, 0

    def _run_cli_container(self, env_name: str, name: str, image: str, **labels: str):
        """Start one CLI shell. The environment, its warm spare, and every
        episode fork share exactly this isolation."""
        return self._docker.containers.run(
            image=image,
            name=name,
            command=["tail", "-f", "/dev/null"],
            detach=True,
            labels={"forge.env": env_name, "forge.managed": "true", "forge.type": "cli", **labels},
            restart_policy={"Name": "unless-stopped"},
            mem_limit=_CLI_MEMORY_LIMIT,
            nano_cpus=_CPU_LIMIT,
            pids_limit=_PID_LIMIT,
            init=True,
            # The agent reaches the shell through `docker exec`, so the
            # container needs no network. Without one, commands see a fixed
            # world instead of whatever the internet serves today.
            network_mode="none",
            environment={
                "TZ": "UTC",
                "PYTHONHASHSEED": _PYTHON_HASH_SEED,
                **_CLI_FAKETIME_ENV,
                "FORGE_DETERMINISM": os.environ.get("FORGE_DETERMINISM", "on"),
            },
        )

    def snapshot_cli(self, env_name: str, container_id: str, run_id: str) -> str:
        """Freeze the environment's shell, including any terminal setup, for a run."""
        tag = cli_snapshot_tag(env_name, run_id)
        _docker_cli("commit", container_id, tag)
        return tag

    @staticmethod
    def discard_cli_snapshot(snapshot: str) -> None:
        """Drop a finished run's snapshot image. A missing image is fine."""
        subprocess.run(["docker", "rmi", "-f", snapshot], capture_output=True, check=False)

    def warm_cli(self, env_name: str, snapshot: str) -> str:
        """Start the spare the next episode takes, so it never waits for a boot."""
        name = f"{self._container_name(env_name)}-warm"
        try:
            self._docker.containers.get(name).remove(force=True)
        except docker.errors.NotFound:
            pass
        container = self._run_cli_container(
            env_name, name, snapshot, **{"forge.role": "warm", "forge.snapshot": snapshot}
        )
        return container.id

    @contextmanager
    def cli_episode(self, env_name: str, snapshot: str, *, refill: bool) -> Iterator[str]:
        """Yield a fresh shell forked from the run's snapshot, then discard it.

        The warm spare is taken when it was forked from this snapshot. After
        the episode, `refill` starts the next spare off the critical path.
        """
        base = self._container_name(env_name)
        episode_name = f"{base}-episode"
        try:
            self._docker.containers.get(episode_name).remove(force=True)
        except docker.errors.NotFound:
            pass
        container = None
        try:
            warm = self._docker.containers.get(f"{base}-warm")
        except docker.errors.NotFound:
            warm = None
        if warm is not None:
            if warm.labels.get("forge.snapshot") == snapshot and warm.status == "running":
                warm.rename(episode_name)
                container = warm
            else:
                warm.remove(force=True)
        if container is None:
            container = self._run_cli_container(
                env_name, episode_name, snapshot, **{"forge.role": "episode"}
            )
        try:
            yield container.id
        finally:
            container.remove(force=True)
            if refill:
                self.warm_cli(env_name, snapshot)

    def run_browser(self, env_name: str) -> tuple[str, int]:
        """Spin up a Chromium+KasmVNC container with no route out.

        Its gateway publishes the web UI and the DevTools port on loopback.
        """
        pull_image(FORGE_BROWSER_IMAGE)
        pull_image(FORGE_GATEWAY_IMAGE)
        self._remove_existing(env_name)
        network = self._sandbox_network(env_name)
        container = self._docker.containers.run(
            image=FORGE_BROWSER_IMAGE,
            name=self._container_name(env_name),
            detach=True,
            network=network.name,
            environment={
                "PUID": "1000",
                "PGID": "1000",
                "TZ": "UTC",
                "FORGE_DETERMINISM": os.environ.get("FORGE_DETERMINISM", "on"),
                # Expose CDP so agents can connect and control the browser programmatically.
                "CHROME_CLI": "--remote-debugging-port=9222 --remote-debugging-address=0.0.0.0 --no-sandbox",
            },
            shm_size="1g",
            labels={"forge.env": env_name, "forge.managed": "true", "forge.type": "browser"},
            restart_policy={"Name": "unless-stopped"},
            mem_limit=_BROWSER_MEMORY_LIMIT,
            nano_cpus=_CPU_LIMIT,
            pids_limit=_PID_LIMIT,
            # No Docker init: the image boots through s6-overlay, which must
            # be PID 1 and reaps zombies itself.
        )
        self._docker.containers.run(
            image=FORGE_GATEWAY_IMAGE,
            name=self._relay_name(env_name),
            entrypoint=["/bin/sh", "-c"],
            command=[
                f"socat TCP-LISTEN:{_BROWSER_CDP_RELAY_PORT},fork,reuseaddr "
                f"TCP:127.0.0.1:{_BROWSER_CDP_PORT}"
            ],
            detach=True,
            # Inside the browser's namespace: it reaches Chromium's loopback
            # and, like the browser, has no route out.
            network_mode=f"container:{container.id}",
            labels={"forge.env": env_name, "forge.managed": "true", "forge.role": "cdp-relay"},
            restart_policy={"Name": "unless-stopped"},
            mem_limit=_GATEWAY_MEMORY_LIMIT,
            nano_cpus=_CPU_LIMIT,
            pids_limit=_PID_LIMIT,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
        )
        gateway = self._start_gateway(
            env_name,
            {_BROWSER_UI_PORT: _BROWSER_UI_PORT, _BROWSER_CDP_PORT: _BROWSER_CDP_RELAY_PORT},
            network,
        )
        port = _wait_for_port_binding(gateway, f"{_BROWSER_UI_PORT}/tcp", attempts=10, interval=0.3)
        return container.id, port

    def run(self, env_name: str, image_tag: str) -> tuple[str, int]:
        pull_image(FORGE_GATEWAY_IMAGE)
        network = self._sandbox_network(env_name)
        container = self._docker.containers.run(
            image=image_tag,
            name=self._container_name(env_name),
            detach=True,
            # No published port and no route out: only the gateway reaches it.
            network=network.name,
            environment={
                "TZ": "UTC",
                "PYTHONHASHSEED": _PYTHON_HASH_SEED,
                "REDIS_URL": self._redis_url,
                "FORGE_ENV_NAME": env_name,
                "FORGE_DETERMINISM": os.environ.get("FORGE_DETERMINISM", "on"),
            },
            labels={"forge.env": env_name, "forge.managed": "true"},
            # `on-failure` with a small retry cap, NOT `unless-stopped`:
            # a buggy LLM-generated app that crashes on boot would otherwise
            # restart forever and the UI would oscillate between "running"
            # and "restarting" without surfacing the actual error. With this
            # policy the container exits cleanly after a few crashes so the
            # GET cross-check can flag it and the user can see logs.
            restart_policy={"Name": "on-failure", "MaximumRetryCount": 3},
            mem_limit=_GENERAL_MEMORY_LIMIT,
            nano_cpus=_CPU_LIMIT,
            pids_limit=_PID_LIMIT,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            init=True,
        )
        gateway = self._start_gateway(env_name, {FORGE_APP_PORT: FORGE_APP_PORT}, network)
        # The port binding is applied asynchronously by the daemon — usually
        # it's there immediately after reload(), but on a busy macOS Docker
        # Desktop it can take a few hundred ms. Poll briefly.
        port = _wait_for_port_binding(gateway, f"{FORGE_APP_PORT}/tcp", attempts=10, interval=0.3)
        return container.id, port

    def stop(self, container_id: str) -> None:
        try:
            container = self._docker.containers.get(container_id)
        except docker.errors.NotFound:
            return
        container.stop(timeout=10)
        for sidecar in self._sidecars_for(container):
            sidecar.stop(timeout=10)

    def start(self, env_name: str, container_id: str, image_tag: str) -> tuple[str, int]:
        """Restart a stopped container, or run a fresh one if it was removed.

        Defensive against the messy states that show up in practice:
          - empty/None container_id (DB reset, partial-state envs) → run fresh
          - container exists but its bound port is gone → run fresh
          - stale stopped container with same forge-<env> name → remove first
          - image was pruned out from under us → clear, actionable error
        """
        existing = None
        if container_id:
            try:
                existing = self._docker.containers.get(container_id)
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError:
                # NullResource and friends: treat the same as NotFound — run fresh.
                pass

        if existing is not None:
            try:
                existing.start()
                existing.reload()
            except docker.errors.APIError:
                # Container exists but won't start (image vanished, etc.). Drop it
                # and fall through to a fresh run.
                try:
                    existing.remove(force=True)
                except docker.errors.APIError:
                    pass
                existing = None
            else:
                if image_tag == "builtin:cli":
                    return existing.id, 0
                # Relay after the browser, since it joins the browser's namespace.
                for sidecar in self._sidecars_for(existing):
                    sidecar.start()
                    sidecar.reload()
                port = self.host_port(existing)
                if port:
                    return existing.id, port
                # Port or gateway disappeared (host reboots, or an app started
                # before gateways existed), so run fresh.
                try:
                    existing.remove(force=True)
                except docker.errors.APIError:
                    pass

        # Fresh-run path — clear any stale forge-<env> container with the same
        # name first, otherwise containers.run raises 409 Conflict.
        self._remove_existing(env_name)
        try:
            if image_tag == "builtin:cli":
                return self.run_cli(env_name)
            elif image_tag == "builtin:browser":
                return self.run_browser(env_name)
            else:
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

    def remove(self, container_id: str, image_tag: str | None = None) -> None:
        self.stop(container_id)
        try:
            container = self._docker.containers.get(container_id)
        except docker.errors.NotFound:
            container = None
        if container is not None:
            sidecars = self._sidecars_for(container)
            container.remove()
            for sidecar in sidecars:
                sidecar.remove(force=True)
            self._remove_network(container.labels.get("forge.env", ""))
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


def ensure_cli_image() -> str:
    """Return the CLI image tag, building it from the pinned base if missing."""
    if _image_cached_locally(CLI_RUNTIME_IMAGE):
        return CLI_RUNTIME_IMAGE
    pull_image(FORGE_CLI_IMAGE)
    try:
        subprocess.run(
            [
                "docker", "build", "--rm",
                "--build-arg", f"BASE_IMAGE={FORGE_CLI_IMAGE}",
                "-t", CLI_RUNTIME_IMAGE,
                str(CLI_DOCKERFILE.parent),
            ],
            check=True, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"CLI image build timed out after {_BUILD_TIMEOUT_S}s") from exc
    except subprocess.CalledProcessError as exc:
        output = (exc.stderr or exc.stdout or "(no output)").strip()
        raise RuntimeError(f"CLI image build failed (exit {exc.returncode}):\n{output}") from exc
    return CLI_RUNTIME_IMAGE


def _docker_cli(*args: str, timeout: float = _BUILD_TIMEOUT_S) -> None:
    """Run one docker CLI command, raising RuntimeError with Docker's output."""
    try:
        subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"docker {args[0]} timed out after {timeout}s") from exc
    except subprocess.CalledProcessError as exc:
        output = (exc.stderr or exc.stdout or "(no output)").strip()
        raise RuntimeError(f"docker {args[0]} failed (exit {exc.returncode}):\n{output}") from exc


def _grader_toolbox_volume() -> str:
    digest = hashlib.sha256(FORGE_GRADER_TOOLBOX_IMAGE.encode()).hexdigest()[:12]
    return f"forge-grader-tools-{digest}"


def ensure_grader_toolbox() -> str:
    """Return the read-only grader toolbox volume, populating it on first use.

    The volume only exists once it is fully populated, so its existence is
    the readiness signal.
    """
    volume = _grader_toolbox_volume()
    inspect = subprocess.run(
        ["docker", "volume", "inspect", volume], capture_output=True, text=True, check=False,
    )
    if inspect.returncode == 0:
        return volume
    pull_image(FORGE_GRADER_TOOLBOX_IMAGE)
    _docker_cli("volume", "create", "--label", "forge.managed=true", volume)
    try:
        # Installed at the path the grader mounts it on, so the applet
        # symlinks stay valid there.
        _docker_cli(
            "run", "--rm", "--network", "none",
            "-v", f"{volume}:{_GRADER_TOOLBOX_DIR}",
            FORGE_GRADER_TOOLBOX_IMAGE, "sh", "-c",
            f"mkdir -p {_GRADER_TOOLBOX_DIR}/bin"
            f" && cp /bin/busybox {_GRADER_TOOLBOX_DIR}/bin/busybox"
            f" && {_GRADER_TOOLBOX_DIR}/bin/busybox --install -s {_GRADER_TOOLBOX_DIR}/bin",
        )
    except RuntimeError as exc:
        subprocess.run(["docker", "volume", "rm", "-f", volume], capture_output=True, check=False)
        raise RuntimeError(f"Could not prepare the grader toolbox: {exc}") from exc
    return volume


@contextmanager
def grading_sandbox(container_id: str) -> Iterator[list[str]]:
    """Yield the command prefix that runs one shell assertion out of the agent's reach.

    The agent's container is committed to an image and a fresh grader starts
    from it: same files, none of the agent's processes, no network. Each
    assertion runs through the read-only toolbox's static shell with a clean
    environment. Append the assertion string to the yielded list.
    """
    toolbox = ensure_grader_toolbox()
    name = f"forge-grade-{container_id[:12]}"
    image = f"{name}:snapshot"
    _docker_cli("commit", container_id, image)
    try:
        _docker_cli(
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
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    finally:
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, check=False)


def prewarm_standard_base_images(
    images: tuple[str, ...] = STANDARD_BASE_IMAGES,
) -> dict[str, str]:
    """Ensure the standard Forge base images are present in the local Docker cache.

    Called from the Celery `worker_ready` signal. Uses `pull_image`, which
    skips already-cached images and falls through Hub mirrors when
    docker.io is unreachable (the original failure mode that motivated this).
    Returns {image: status} where status is "cached", "pulled", or "failed: <msg>".

    This is the primary defence against Docker Hub flakiness: by the time
    user-triggered build_sandbox_task runs, the canonical Python base is
    already local, so `ContainerRuntime.build()` short-circuits the pull
    and never touches any registry on the hot path.
    """
    import logging
    log = logging.getLogger(__name__)
    results: dict[str, str] = {}
    for image in images:
        if _image_cached_locally(image):
            results[image] = "cached"
            log.info("[prewarm] %s already cached", image)
            continue
        try:
            pull_image(image)
            results[image] = "pulled"
            log.info("[prewarm] %s pulled successfully", image)
        except RuntimeError as exc:
            # Don't crash the worker if every registry is down at boot —
            # pull_image will retry when a user request arrives.
            results[image] = f"failed: {exc}"
            log.warning("[prewarm] %s pull failed: %s", image, exc)
    return results
