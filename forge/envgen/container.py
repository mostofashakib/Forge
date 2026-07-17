from __future__ import annotations
import os
import re
import subprocess
import time
from pathlib import Path
import docker
import docker.errors

from forge.logging_utils import redact_sensitive_text

from forge.runtime.network_isolation import check_generated_env
from forge.settings import redis_url


# Single canonical base image for every generated env. The LLM-generated
# Dockerfile gets its FROM line normalised to this, so all builds depend
# on exactly one image. We pre-warm it at worker startup, which means the
# user-triggered build path always finds it cached and never contacts
# Docker Hub — making the system immune to transient Hub EOF outages.
FORGE_PYTHON_BASE = os.environ.get("FORGE_PYTHON_BASE_IMAGE", "python:3.12-slim")
FORGE_CLI_IMAGE = os.environ.get("FORGE_CLI_IMAGE", "ubuntu:22.04")
FORGE_BROWSER_IMAGE = os.environ.get(
    "FORGE_BROWSER_IMAGE", "lscr.io/linuxserver/chromium:latest"
)

# Standard base images Forge needs locally. Pre-warmed at Celery worker
# startup so user-triggered builds never wait on Hub.
STANDARD_BASE_IMAGES: tuple[str, ...] = (
    FORGE_PYTHON_BASE,
    FORGE_CLI_IMAGE,
    FORGE_BROWSER_IMAGE,
)

_DOCKERFILE = f"""\
FROM {FORGE_PYTHON_BASE}
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn sqlalchemy redis httpx
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

# Runtime limits are intentionally conservative defaults for generated code.
# They can be overridden for larger local experiments without changing code.
_GENERAL_MEMORY_LIMIT = os.environ.get("FORGE_CONTAINER_MEMORY", "1g")
_BROWSER_MEMORY_LIMIT = os.environ.get("FORGE_BROWSER_MEMORY", "2g")
_CLI_MEMORY_LIMIT = os.environ.get("FORGE_CLI_MEMORY", "1g")
_CPU_LIMIT = int(os.environ.get("FORGE_CONTAINER_NANO_CPUS", "1000000000"))
_PID_LIMIT = int(os.environ.get("FORGE_CONTAINER_PIDS", "256"))


def _loopback_port() -> tuple[str, None]:
    """Ask Docker for a random host port bound only to loopback."""
    return ("127.0.0.1", None)

# Every Forge-generated FastAPI app needs these at runtime. The LLM that
# writes requirements.txt is a different model (Haiku) than the one that
# writes main.py (Sonnet), and they drift — Sonnet routinely emits
# `import redis` while Haiku forgets to list it. The container then
# crashes on boot with ModuleNotFoundError. Inject this baseline at build
# time so the runtime imports always resolve, regardless of what the LLM
# happened to put in requirements.txt.
_BASELINE_REQUIREMENTS: tuple[str, ...] = (
    "fastapi",
    "uvicorn[standard]",
    "sqlalchemy",
    "redis",
    "httpx",
    "python-multipart",
    "pydantic",
)


def _existing_packages(req_text: str) -> set[str]:
    """Lower-cased set of package names already declared (extras/version stripped)."""
    out: set[str] = set()
    for line in req_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip extras "[standard]" and version specifiers "==1.2", ">=1.0", etc.
        name = line.split("[", 1)[0]
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            name = name.split(sep, 1)[0]
        out.add(name.strip().lower())
    return out


def _normalise_requirements(app_dir: Path) -> bool:
    """Ensure the FastAPI + Forge baseline deps are present in requirements.txt.

    Returns True if the file was created or amended.
    """
    req_file = app_dir / "requirements.txt"
    if not req_file.exists():
        req_file.write_text("\n".join(_BASELINE_REQUIREMENTS) + "\n")
        return True

    existing = req_file.read_text()
    declared = _existing_packages(existing)
    missing = [
        dep for dep in _BASELINE_REQUIREMENTS
        if dep.split("[", 1)[0].lower() not in declared
    ]
    if not missing:
        return False
    req_file.write_text(existing.rstrip() + "\n" + "\n".join(missing) + "\n")
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
            _docker_tag(mirror_ref, image)
            log.info("[pull] %s served by %s", image, mirror)
            return
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            errors.append(f"{mirror} → {redact_sensitive_text(exc)}")
            log.info("[pull] %s unavailable for %s", mirror, image)

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
    def __init__(self) -> None:
        self._docker_client: docker.DockerClient | None = None
        self._redis_url = redis_url()

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
        else:
            # Normalise any LLM-generated `FROM python:X` to the single
            # canonical Forge base. This guarantees every build depends
            # on the one image we pre-warm, so Hub is never on the hot path.
            _normalise_dockerfile_base(dockerfile)
            # Force EXPOSE / --port to FORGE_APP_PORT so the in-container
            # listener and the host-side port mapping never disagree, no
            # matter what port the LLM picked.
            _normalise_dockerfile_port(dockerfile)

        # Inject the baseline runtime deps. The LLM that writes main.py
        # (Sonnet) and the one that writes requirements.txt (Haiku) are
        # separate models and drift — main.py routinely imports `redis`,
        # `httpx`, etc. while requirements.txt forgets them, which crashes
        # the container at boot with ModuleNotFoundError.
        _normalise_requirements(app_dir)

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
            )
        except subprocess.CalledProcessError as exc:
            output = (exc.stderr or exc.stdout or "(no output)").strip()
            raise RuntimeError(
                f"docker build failed (exit {exc.returncode}):\n{output}"
            ) from exc
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
        pull_image(FORGE_CLI_IMAGE)
        self._remove_existing(env_name)
        container = self._docker.containers.run(
            image=FORGE_CLI_IMAGE,
            name=self._container_name(env_name),
            command=["tail", "-f", "/dev/null"],
            detach=True,
            labels={"forge.env": env_name, "forge.managed": "true", "forge.type": "cli"},
            restart_policy={"Name": "unless-stopped"},
            mem_limit=_CLI_MEMORY_LIMIT,
            nano_cpus=_CPU_LIMIT,
            pids_limit=_PID_LIMIT,
            init=True,
        )
        return container.id, 0

    def run_browser(self, env_name: str) -> tuple[str, int]:
        """Spin up a Chromium+KasmVNC container, exposing the web UI on a random port."""
        pull_image(FORGE_BROWSER_IMAGE)
        self._remove_existing(env_name)
        container = self._docker.containers.run(
            image=FORGE_BROWSER_IMAGE,
            name=self._container_name(env_name),
            detach=True,
            ports={"3000/tcp": _loopback_port(), "9222/tcp": _loopback_port()},
            environment={
                "PUID": "1000",
                "PGID": "1000",
                "TZ": "UTC",
                # Expose CDP so agents can connect and control the browser programmatically.
                "CHROME_CLI": "--remote-debugging-port=9222 --remote-debugging-address=0.0.0.0 --no-sandbox",
            },
            shm_size="1g",
            labels={"forge.env": env_name, "forge.managed": "true", "forge.type": "browser"},
            restart_policy={"Name": "unless-stopped"},
            mem_limit=_BROWSER_MEMORY_LIMIT,
            nano_cpus=_CPU_LIMIT,
            pids_limit=_PID_LIMIT,
            init=True,
        )
        port = _wait_for_port_binding(container, "3000/tcp", attempts=10, interval=0.3)
        return container.id, port

    def run(self, env_name: str, image_tag: str) -> tuple[str, int]:
        container = self._docker.containers.run(
            image=image_tag,
            name=self._container_name(env_name),
            detach=True,
            ports={"8000/tcp": _loopback_port()},
            environment={"REDIS_URL": self._redis_url, "FORGE_ENV_NAME": env_name},
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
        # The port binding is applied asynchronously by the daemon — usually
        # it's there immediately after reload(), but on a busy macOS Docker
        # Desktop it can take a few hundred ms. Poll briefly.
        port = _wait_for_port_binding(container, "8000/tcp", attempts=10, interval=0.3)
        return container.id, port

    def stop(self, container_id: str) -> None:
        try:
            container = self._docker.containers.get(container_id)
            container.stop(timeout=10)
        except docker.errors.NotFound:
            pass

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
                port_key = "3000/tcp" if image_tag == "builtin:browser" else "8000/tcp"
                bindings = existing.ports.get(port_key) or []
                if bindings:
                    return existing.id, int(bindings[0]["HostPort"])
                # Port disappeared (rare but happens after host reboots) — run fresh.
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
