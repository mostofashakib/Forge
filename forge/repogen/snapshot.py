"""Take a repo at a pinned commit and make its files safe to build and train on.

Downloads and archives are untrusted: the download is size-capped, and
extraction goes through tarfile's `data` filter, which refuses absolute paths,
`..` escapes, links that leave the destination, and device files. Every
remaining source file passes the safety gate as internet data.
"""
from __future__ import annotations

import os
import re
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from forge.safety.service import DatasetSafetyService

_SHA = re.compile(r"^[0-9a-f]{40}$")
_CHUNK = 1 << 16
_TIMEOUT_S = 120.0
_MAX_FILE_BYTES = 1_000_000
_BINARY_PROBE = 8192
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "vendor", "dist", "build", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache",
})
# Lockfiles stay on disk for installs. They hold hashes, not code, so they are not screened.
_LOCKFILES = frozenset({"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock", "uv.lock", "Pipfile.lock"})
_GENERATED = re.compile(r"\.(?:min\.js|min\.css|map)$")


class SnapshotError(RuntimeError):
    """The repo could not be downloaded or extracted safely."""


@dataclass(frozen=True)
class RepoVerdict:
    accepted: bool
    # Relative path to the rules that flagged it.
    rejections: dict[str, list[str]] = field(default_factory=dict)
    replacements: dict[str, int] = field(default_factory=dict)


def fetch_tarball(
    full_name: str, sha: str, token: str, dest: Path, *, max_bytes: int, http: httpx.Client | None = None
) -> Path:
    """Download the repo at commit `sha` to `dest`, refusing anything over `max_bytes`."""
    if not _SHA.match(sha):
        raise ValueError(f"expected a 40-character commit SHA, got {sha!r}")
    client = http or httpx.Client(follow_redirects=True, timeout=_TIMEOUT_S)
    url = f"https://api.github.com/repos/{full_name}/tarball/{sha}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code != 200:
                raise SnapshotError(f"downloading {full_name}@{sha[:12]} failed: HTTP {response.status_code}")
            written = 0
            with dest.open("wb") as handle:
                for chunk in response.iter_bytes(_CHUNK):
                    written += len(chunk)
                    if written > max_bytes:
                        raise SnapshotError(f"{full_name}@{sha[:12]} is larger than {max_bytes} bytes")
                    handle.write(chunk)
    except SnapshotError:
        dest.unlink(missing_ok=True)
        raise
    finally:
        if http is None:
            client.close()
    return dest


def safe_extract(tarball: Path, dest: Path, *, max_bytes: int) -> Path:
    """Extract a GitHub tarball into `dest` without its top folder. Hostile archives raise."""
    with tarfile.open(tarball, "r:gz") as tar:
        members = tar.getmembers()
        tops = {member.name.split("/", 1)[0] for member in members}
        if len(tops) != 1 or "" in tops:
            raise SnapshotError(f"expected one top-level folder, found {sorted(tops)}")
        if sum(member.size for member in members if member.isfile()) > max_bytes:
            raise SnapshotError(f"archive expands past {max_bytes} bytes")
        prefix = tops.pop() + "/"
        stripped = []
        for member in members:
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix):]
            if member.islnk():
                member.linkname = member.linkname.removeprefix(prefix)
            stripped.append(member)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            tar.extractall(dest, members=stripped, filter="data")
        except tarfile.FilterError as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise SnapshotError(f"unsafe archive member: {exc}") from exc
    return dest


def select_files(root: Path) -> list[Path]:
    """Source files worth screening, relative to `root`: no deps, builds, binaries, or bundles."""
    selected: list[Path] = []
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in _SKIP_DIRS)
        for name in sorted(names):
            path = Path(directory) / name
            if name in _LOCKFILES or _GENERATED.search(name) or path.is_symlink():
                continue
            if path.stat().st_size > _MAX_FILE_BYTES or _is_binary(path):
                continue
            selected.append(path.relative_to(root))
    return selected


def screen_repo(root: Path, files: list[Path], safety: DatasetSafetyService) -> RepoVerdict:
    """Screen every file. Scrubbed text is written back. Any flagged file rejects the repo."""
    rejections: dict[str, list[str]] = {}
    replacements: Counter[str] = Counter()
    for rel in files:
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        result = safety.screen({"path": str(rel), "content": text})
        if not result.kept:
            rejections[str(rel)] = sorted({finding.rule for finding in result.findings})
            continue
        replacements.update(result.replacements)
        if result.record["content"] != text:
            path.write_text(result.record["content"], encoding="utf-8")
    return RepoVerdict(not rejections, rejections, dict(replacements))


def _is_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\x00" in handle.read(_BINARY_PROBE)
