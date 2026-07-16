"""Pull a Docker image via the OCI registry HTTP API directly.

This bypasses dockerd's pull pipeline entirely — useful when the Docker
daemon's HTTP/2 client is unstable on the current network (MTU mismatch,
broken IPv6 routing, idle-stream resets), which surfaces as repeated
`EOF` errors against unrelated registries (Hub CDN, AWS CloudFront,
Google's mirror) all at once. Forcing HTTP/1.1 over a fresh httpx client
sidesteps that whole class of failures.

Flow: GET manifest list → pick platform → GET manifest → stream config
and layer blobs to a temp dir → assemble a Docker-load-format tar →
`docker load`. After load the image is in the local cache under the
canonical name and the rest of the system can use it transparently.

Currently supports Docker Hub references only (the token-auth flow is
Hub-specific). Non-Hub images keep using `docker pull`.
"""
from __future__ import annotations

import gzip
import json
import logging
import platform as _platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import httpx

from forge.logging_utils import install_sensitive_log_filter, redact_sensitive_text

log = logging.getLogger(__name__)
install_sensitive_log_filter("httpx", "httpcore", __name__)

_DOCKER_HUB_REGISTRY = "registry-1.docker.io"
_DOCKER_HUB_AUTH_URL = "https://auth.docker.io/token"
_DOCKER_HUB_AUTH_SERVICE = "registry.docker.io"

# httpx settings — explicit HTTP/1.1, generous read window for slow CDN
# edges, modest connect timeout so we fail fast on dead routes.
_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
_HTTP_LIMITS = httpx.Limits(max_keepalive_connections=4, max_connections=8)

# OCI / Docker manifest media types we accept
_ACCEPT_MANIFEST = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])


def _detect_platform() -> tuple[str, str]:
    """Return (os, arch) in OCI form for the current host."""
    arch_map = {
        "x86_64": "amd64", "amd64": "amd64",
        "arm64": "arm64", "aarch64": "arm64",
        "armv7l": "arm",
    }
    arch = arch_map.get(_platform.machine().lower(), "amd64")
    return ("linux", arch)


def _parse_image_ref(image: str) -> tuple[str, str, str]:
    """Split a reference like 'python:3.12-slim' → (registry, repo, tag).

    Bare names get the implicit 'library/' Hub prefix and the 'latest'
    tag default, matching dockerd's resolution rules.
    """
    registry = _DOCKER_HUB_REGISTRY
    rest = image
    if "/" in image:
        head, tail = image.split("/", 1)
        # An explicit registry component contains a dot (DNS), a port colon,
        # or is exactly 'localhost' — otherwise it's a Hub user namespace.
        if "." in head or ":" in head or head == "localhost":
            registry = head
            rest = tail
    if ":" in rest:
        repo, tag = rest.rsplit(":", 1)
    else:
        repo, tag = rest, "latest"
    if registry == _DOCKER_HUB_REGISTRY and "/" not in repo:
        repo = f"library/{repo}"
    return registry, repo, tag


def _get_hub_token(repo: str, client: httpx.Client) -> str:
    """Anonymous pull-scoped token from Docker Hub's auth endpoint."""
    r = client.get(
        _DOCKER_HUB_AUTH_URL,
        params={"service": _DOCKER_HUB_AUTH_SERVICE, "scope": f"repository:{repo}:pull"},
    )
    r.raise_for_status()
    return r.json()["token"]


def _select_platform_digest(index: dict, target_os: str, target_arch: str) -> str:
    """Pick the manifest digest for our (os, arch) from a manifest index."""
    for m in index.get("manifests", []):
        plat = m.get("platform", {})
        if plat.get("os") == target_os and plat.get("architecture") == target_arch:
            return m["digest"]
    available = [m.get("platform") for m in index.get("manifests", [])]
    raise RuntimeError(
        f"No manifest for {target_os}/{target_arch} in image index "
        f"(available: {available})"
    )


def _download_blob(
    client: httpx.Client,
    base_url: str,
    headers: dict,
    digest: str,
    dest: Path,
) -> None:
    """Stream a blob to disk in 64 KiB chunks (avoids loading layers into RAM)."""
    with client.stream("GET", f"{base_url}/blobs/{digest}", headers=headers) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=64 * 1024):
                f.write(chunk)


def _pull_via_http(image: str) -> None:
    """Pull a Docker Hub image via direct HTTPS, then `docker load` it locally.

    Raises RuntimeError on any failure (caller is expected to fall through
    to other transports or surface the error).
    """
    registry, repo, tag = _parse_image_ref(image)
    if registry != _DOCKER_HUB_REGISTRY:
        raise RuntimeError(
            f"pull_via_http only supports Docker Hub references; got {image!r}"
        )

    target_os, target_arch = _detect_platform()
    base_url = f"https://{registry}/v2/{repo}"

    with httpx.Client(
        timeout=_HTTP_TIMEOUT,
        limits=_HTTP_LIMITS,
        http2=False,
        follow_redirects=True,
    ) as client:
        token = _get_hub_token(repo, client)
        headers = {"Authorization": f"Bearer {token}", "Accept": _ACCEPT_MANIFEST}

        log.info("[http-pull] manifest %s:%s", repo, tag)
        r = client.get(f"{base_url}/manifests/{tag}", headers=headers)
        r.raise_for_status()
        manifest_or_index = r.json()

        # Resolve a manifest list/index down to a single platform manifest
        if "manifests" in manifest_or_index:
            digest = _select_platform_digest(manifest_or_index, target_os, target_arch)
            log.info("[http-pull] selected %s/%s manifest %s",
                     target_os, target_arch, digest[:19])
            r = client.get(f"{base_url}/manifests/{digest}", headers=headers)
            r.raise_for_status()
            manifest = r.json()
        else:
            manifest = manifest_or_index

        config = manifest["config"]
        layers = manifest["layers"]

        with tempfile.TemporaryDirectory(prefix="forge-http-pull-") as tmpdir:
            tmp = Path(tmpdir)

            # Config blob → <hash>.json
            config_filename = config["digest"].replace(":", "_") + ".json"
            log.info("[http-pull] config %s", config["digest"][:19])
            _download_blob(client, base_url, headers, config["digest"],
                           tmp / config_filename)

            # Layer blobs → <hash>/layer.tar (decompressed if gzipped)
            layer_paths: list[str] = []
            for i, layer in enumerate(layers, 1):
                layer_dir_name = layer["digest"].replace(":", "_")
                layer_dir = tmp / layer_dir_name
                layer_dir.mkdir()
                gz_path = layer_dir / "layer.tar.gz"
                final_path = layer_dir / "layer.tar"

                size_mb = layer.get("size", 0) / (1024 * 1024)
                log.info("[http-pull] layer %d/%d (%s, %.1f MiB)",
                         i, len(layers), layer["digest"][:19], size_mb)
                _download_blob(client, base_url, headers, layer["digest"], gz_path)

                if "gzip" in layer.get("mediaType", "").lower():
                    with gzip.open(gz_path, "rb") as gz, open(final_path, "wb") as out:
                        shutil.copyfileobj(gz, out, length=64 * 1024)
                    gz_path.unlink()
                else:
                    gz_path.rename(final_path)

                layer_paths.append(f"{layer_dir_name}/layer.tar")

            # Docker-load manifest (legacy format dockerd accepts)
            (tmp / "manifest.json").write_text(json.dumps([{
                "Config": config_filename,
                "RepoTags": [image],
                "Layers": layer_paths,
            }]))

            out_tar = tmp / "_image.tar"
            with tarfile.open(out_tar, "w") as tar:
                for entry in tmp.iterdir():
                    if entry.name != "_image.tar":
                        tar.add(entry, arcname=entry.name)

            log.info("[http-pull] docker load %s", image)
            subprocess.run(
                ["docker", "load", "-i", str(out_tar)],
                check=True, capture_output=True, text=True,
            )


def pull_via_http(image: str) -> None:
    """Run the HTTP pull without ever propagating signed URLs or credentials."""

    try:
        _pull_via_http(image)
    except Exception as exc:
        raise RuntimeError(redact_sensitive_text(exc)) from None
