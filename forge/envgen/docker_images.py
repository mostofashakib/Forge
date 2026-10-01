"""Which images Forge runs, what their tags are, and how they reach the local cache.

Every image is pulled through `pull_image`, which skips cached images and
falls back through Hub mirrors and direct HTTPS when docker.io is down.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from forge.envgen.docker_cli import BUILD_TIMEOUT_S
from forge.logging_utils import redact_sensitive_text

logger = logging.getLogger(__name__)


# Single canonical base image for every generated env. The LLM-generated
# Dockerfile gets its FROM line normalised to this, so all builds depend
# on exactly one image. We pre-warm it at worker startup, which means the
# user-triggered build path always finds it cached and never contacts
# Docker Hub — making the system immune to transient Hub EOF outages.
#
# Every image is pinned by digest. A tag moves when upstream republishes it,
# so the same environment would otherwise build on different bytes each week.
DEFAULT_PYTHON_BASE_IMAGE = (
    "python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f"
)
DEFAULT_CLI_IMAGE = "ubuntu:22.04@sha256:b8b6ee6aa931ecd9d0d952abc34dc0e5f7c6a30c6bb71b079fe399fde0329c02"
DEFAULT_BROWSER_IMAGE = (
    "lscr.io/linuxserver/chromium:latest"
    "@sha256:cf6200ccdcb224feaf5d3bde4ce45b3783c926a7496c98059cae1e0db78e5b2f"
)
FORGE_PYTHON_BASE = os.environ.get("FORGE_PYTHON_BASE_IMAGE", DEFAULT_PYTHON_BASE_IMAGE)
FORGE_CLI_IMAGE = os.environ.get("FORGE_CLI_IMAGE", DEFAULT_CLI_IMAGE)
FORGE_BROWSER_IMAGE = os.environ.get("FORGE_BROWSER_IMAGE", DEFAULT_BROWSER_IMAGE)

FORGE_GATEWAY_IMAGE = os.environ.get(
    "FORGE_GATEWAY_IMAGE",
    "alpine/socat:1.8.0.3@sha256:beb4a68d9e4fe6b0f21ea774a0fde6c31f580dde6368939ed70100c5385b015e",
)

# CLI grading uses only this toolbox: a static BusyBox from a pinned image,
# copied into a volume and mounted read-only into the grader. Static binaries
# ignore LD_PRELOAD and /etc/ld.so.preload, so nothing the agent wrote in its
# container can change what an assertion's `grep` or `test` reports.
FORGE_GRADER_TOOLBOX_IMAGE = os.environ.get(
    "FORGE_GRADER_TOOLBOX_IMAGE",
    "busybox:1.37.0-musl@sha256:5cec3fc171c87218698e85a52af7087de727372aae264a787b8112901a5b0092",
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


def initial_snapshot_tag(env_name: str, seed: int = 0) -> str:
    """The immutable Docker image snapshot of initial state with seed data."""
    repo = re.sub(r"[^a-z0-9.-]", "-", env_name.lower().replace("_", "-"))
    return f"forge-snapshot-{repo}:seed-{seed}"


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


def image_cached_locally(image: str) -> bool:
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
    log = logger

    # Already cached locally — nothing to do.
    if image_cached_locally(image):
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



def ensure_cli_image() -> str:
    """Return the CLI image tag, building it from the pinned base if missing."""
    if image_cached_locally(CLI_RUNTIME_IMAGE):
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
            check=True, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"CLI image build timed out after {BUILD_TIMEOUT_S}s") from exc
    except subprocess.CalledProcessError as exc:
        output = (exc.stderr or exc.stdout or "(no output)").strip()
        raise RuntimeError(f"CLI image build failed (exit {exc.returncode}):\n{output}") from exc
    return CLI_RUNTIME_IMAGE



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
    log = logger
    results: dict[str, str] = {}
    for image in images:
        if image_cached_locally(image):
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
