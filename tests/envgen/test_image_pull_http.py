"""Tests for forge.envgen._image_pull_http — direct OCI registry pull via httpx.

These tests cover the parsing/orchestration logic. The actual httpx wire
calls are mocked — the goal is to verify the protocol flow (auth → manifest
list → platform manifest → blobs → docker load) is wired correctly, and
that the temp-tar layout matches what `docker load` expects.
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.envgen._image_pull_http import (
    _detect_platform,
    _parse_image_ref,
    _select_platform_digest,
    pull_via_http,
)


# ---------------------------------------------------------------------------
# _parse_image_ref
# ---------------------------------------------------------------------------

def test_parse_bare_repo_gets_library_prefix_and_default_tag():
    assert _parse_image_ref("python") == ("registry-1.docker.io", "library/python", "latest")


def test_parse_bare_repo_with_tag():
    assert _parse_image_ref("python:3.12-slim") == (
        "registry-1.docker.io", "library/python", "3.12-slim",
    )


def test_parse_user_repo_with_tag():
    """Hub user images don't get the library/ prefix."""
    assert _parse_image_ref("user/image:tag") == (
        "registry-1.docker.io", "user/image", "tag",
    )


def test_parse_explicit_registry_with_dot():
    assert _parse_image_ref("ghcr.io/user/image:tag") == (
        "ghcr.io", "user/image", "tag",
    )


def test_parse_explicit_registry_with_port():
    assert _parse_image_ref("localhost:5000/foo:bar") == (
        "localhost:5000", "foo", "bar",
    )


# ---------------------------------------------------------------------------
# _detect_platform
# ---------------------------------------------------------------------------

def test_detect_platform_returns_linux():
    """We always pull linux images regardless of host OS — sandboxes run linux."""
    os_, arch = _detect_platform()
    assert os_ == "linux"
    assert arch in {"amd64", "arm64", "arm"}


# ---------------------------------------------------------------------------
# _select_platform_digest
# ---------------------------------------------------------------------------

def test_select_platform_digest_picks_matching_entry():
    index = {
        "manifests": [
            {"digest": "sha256:aaa", "platform": {"os": "linux", "architecture": "arm64"}},
            {"digest": "sha256:bbb", "platform": {"os": "linux", "architecture": "amd64"}},
            {"digest": "sha256:ccc", "platform": {"os": "windows", "architecture": "amd64"}},
        ]
    }
    assert _select_platform_digest(index, "linux", "amd64") == "sha256:bbb"


def test_select_platform_digest_raises_when_no_match():
    index = {
        "manifests": [
            {"digest": "sha256:aaa", "platform": {"os": "windows", "architecture": "amd64"}},
        ]
    }
    with pytest.raises(RuntimeError, match="No manifest for linux/amd64"):
        _select_platform_digest(index, "linux", "amd64")


# ---------------------------------------------------------------------------
# pull_via_http — orchestration with mocked httpx + subprocess
# ---------------------------------------------------------------------------

def _gzip_bytes(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(data)
    return buf.getvalue()


def _make_streaming_response(body: bytes, status_code: int = 200) -> MagicMock:
    """A mock that mimics httpx.Client.stream() returning chunked bytes."""
    cm = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_bytes = MagicMock(return_value=iter([body]))
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _make_json_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


def test_pull_via_http_rejects_non_hub_image():
    """Caller is responsible for ensuring this is a Hub reference."""
    with pytest.raises(RuntimeError, match="only supports Docker Hub"):
        pull_via_http("ghcr.io/user/image:tag")


def test_pull_via_http_full_flow_with_manifest_list():
    """End-to-end: token → manifest list → platform manifest → config + layer → docker load."""
    real_layer_bytes = b"this would be a layer.tar in real life"
    gzipped_layer = _gzip_bytes(real_layer_bytes)

    manifest_list = {
        "manifests": [
            {"digest": "sha256:platdigest", "platform": {"os": "linux", "architecture": "amd64"}},
        ],
    }
    platform_manifest = {
        "config": {"digest": "sha256:cfg123", "size": 100,
                   "mediaType": "application/vnd.docker.container.image.v1+json"},
        "layers": [
            {"digest": "sha256:lyr456", "size": len(gzipped_layer),
             "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip"},
        ],
    }
    config_bytes = b'{"architecture":"amd64","os":"linux"}'

    # Build the httpx.Client mock with branching responses
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)

    def get_side_effect(url, **_):
        if url == "https://auth.docker.io/token":
            return _make_json_response({"token": "fake-bearer"})
        if url.endswith("/manifests/3.12-slim"):
            return _make_json_response(manifest_list)
        if url.endswith("/manifests/sha256:platdigest"):
            return _make_json_response(platform_manifest)
        raise AssertionError(f"unexpected GET {url}")

    def stream_side_effect(_method, url, **_):
        if url.endswith("/blobs/sha256:cfg123"):
            return _make_streaming_response(config_bytes)
        if url.endswith("/blobs/sha256:lyr456"):
            return _make_streaming_response(gzipped_layer)
        raise AssertionError(f"unexpected stream {url}")

    client.get = MagicMock(side_effect=get_side_effect)
    client.stream = MagicMock(side_effect=stream_side_effect)

    # Capture the tar bytes inside the mock — the real temp dir gets
    # cleaned up the moment pull_via_http returns.
    captured: dict = {}

    def capture_docker_load(cmd, **_):
        if cmd[:3] == ["docker", "load", "-i"]:
            captured["tar_bytes"] = Path(cmd[3]).read_bytes()
        return MagicMock(returncode=0)

    with patch("forge.envgen._image_pull_http.httpx.Client", return_value=client), \
         patch("forge.envgen._image_pull_http._detect_platform", return_value=("linux", "amd64")), \
         patch("forge.envgen._image_pull_http.subprocess.run", side_effect=capture_docker_load):
        pull_via_http("python:3.12-slim")

    assert "tar_bytes" in captured, "docker load was not invoked"

    with tarfile.open(fileobj=io.BytesIO(captured["tar_bytes"]), mode="r") as tar:
        names = set(tar.getnames())
        assert "manifest.json" in names
        assert any(n.endswith(".json") and "cfg123" in n for n in names)
        assert any(n.endswith("layer.tar") for n in names)

        manifest_member = tar.extractfile("manifest.json")
        assert manifest_member is not None
        manifest_json = json.loads(manifest_member.read())
        assert manifest_json[0]["RepoTags"] == ["python:3.12-slim"]
        assert manifest_json[0]["Layers"][0].endswith("/layer.tar")

        # The gzipped layer was decompressed before being written into the tar
        layer_member = tar.extractfile(manifest_json[0]["Layers"][0])
        assert layer_member is not None
        assert layer_member.read() == real_layer_bytes


def test_pull_via_http_handles_single_manifest_no_index():
    """Older images return a manifest directly (no manifest list). Should still work."""
    layer_bytes = b"layer content"
    gzipped = _gzip_bytes(layer_bytes)

    direct_manifest = {
        "config": {"digest": "sha256:cfg", "size": 10,
                   "mediaType": "application/vnd.docker.container.image.v1+json"},
        "layers": [
            {"digest": "sha256:lyr", "size": len(gzipped),
             "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip"},
        ],
    }

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)

    def get_side_effect(url, **_):
        if "/token" in url:
            return _make_json_response({"token": "t"})
        if "/manifests/" in url:
            return _make_json_response(direct_manifest)
        raise AssertionError(url)

    def stream_side_effect(_method, url, **_):
        if "cfg" in url:
            return _make_streaming_response(b"{}")
        if "lyr" in url:
            return _make_streaming_response(gzipped)
        raise AssertionError(url)

    client.get = MagicMock(side_effect=get_side_effect)
    client.stream = MagicMock(side_effect=stream_side_effect)

    with patch("forge.envgen._image_pull_http.httpx.Client", return_value=client), \
         patch("forge.envgen._image_pull_http.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        pull_via_http("python:3.12-slim")

    # Should NOT make a second manifest GET (no platform list to resolve)
    manifest_calls = [c for c in client.get.call_args_list if "/manifests/" in c.args[0]]
    assert len(manifest_calls) == 1
    # And docker load was still called
    assert mock_run.call_count == 1
    assert mock_run.call_args.args[0][:2] == ["docker", "load"]
