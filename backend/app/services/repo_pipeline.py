"""Shortlists, approval, and per-repo builds for GitHub repo environments.

Approval turns shortlisted repos into queued `repo` sandboxes, within the same
environment cap every sandbox counts against. A build snapshots the repo,
screens it as internet data, builds it, and launches it. Each repo's outcome
is recorded on its own row, so one bad repo never stops the others.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.models import RepoEnvironment, RepoShortlist, SandboxEnvironment
from backend.app.services.sandbox_capacity import active_sandbox_count
from forge.repogen.build import RepoBuildResult
from forge.repogen.github_mcp import GitHubSource
from forge.repogen.snapshot import screen_repo, select_files
from forge.repogen.types import DiscoveryCriteria, Shortlist
from forge.safety.service import DatasetSafetyService

logger = logging.getLogger(__name__)

REPO_TTL_DAYS = 30
_STALE_SANDBOX = {"deleted", "expired"}


class ShortlistNotFound(LookupError):
    pass


class NotInShortlist(ValueError):
    pass


class AlreadyApproved(ValueError):
    pass


class CapacityExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class RepoBuildDeps:
    """What a build needs from the outside world, so tests can stand in for it."""

    # (full_name, sha, empty work dir) -> extracted repo root
    download: Callable[[str, str, Path], Path]
    safety: DatasetSafetyService
    # (env_name, repo root, GitHub language) -> result
    build: Callable[[str, Path, str | None], RepoBuildResult]
    # (env_name, image tag) -> container id
    launch: Callable[[str, str], str]
    work_root: Path


def repo_env_name(full_name: str) -> str:
    return "repo_" + re.sub(r"[^a-z0-9]+", "_", full_name.lower()).strip("_")


def save_shortlist(db: Session, shortlist: Shortlist, criteria: DiscoveryCriteria) -> RepoShortlist:
    row = RepoShortlist(
        id=f"rs_{uuid.uuid4().hex[:12]}",
        brief=shortlist.brief,
        criteria=json.dumps(asdict(criteria)),
        candidates=json.dumps([
            {**asdict(ranked.candidate), "pushed_at": _iso(ranked.candidate.pushed_at),
             "fit": ranked.fit, "reason": ranked.reason}
            for ranked in shortlist.repos
        ]),
        rejected=json.dumps(shortlist.rejected),
    )
    db.add(row)
    db.commit()
    return row


def shortlist_view(row: RepoShortlist) -> dict:
    return {
        "id": row.id,
        "brief": row.brief,
        "criteria": json.loads(row.criteria),
        "candidates": json.loads(row.candidates),
        "rejected": json.loads(row.rejected),
    }


def approve(
    db: Session, shortlist_id: str, full_names: Sequence[str], source: GitHubSource, *, limit: int
) -> list[RepoEnvironment]:
    """Queue one `repo` sandbox per approved repo, pinned to its branch head."""
    row = db.get(RepoShortlist, shortlist_id)
    if row is None:
        raise ShortlistNotFound(shortlist_id)
    candidates = {c["full_name"]: c for c in json.loads(row.candidates)}
    unknown = [name for name in full_names if name not in candidates]
    if unknown:
        raise NotInShortlist(f"not in shortlist {shortlist_id}: {', '.join(unknown)}")
    existing = [name for name in full_names if _live(db, repo_env_name(name))]
    if existing:
        raise AlreadyApproved(f"already approved: {', '.join(existing)}")
    if active_sandbox_count(db) + len(full_names) > limit:
        raise CapacityExceeded(f"approving {len(full_names)} repos would pass the {limit}-environment limit")

    pinned = {name: source.head_sha(name, candidates[name]["default_branch"]) for name in full_names}
    for name in full_names:
        _clear_stale(db, repo_env_name(name))
    expires_at = datetime.now(timezone.utc) + timedelta(days=REPO_TTL_DAYS)
    envs = []
    for name in full_names:
        env_name = repo_env_name(name)
        db.add(SandboxEnvironment(
            id=env_name, status="queued", env_type="repo", ttl_days=REPO_TTL_DAYS,
            expires_at=expires_at, has_ui=False,
        ))
        env = RepoEnvironment(
            env_name=env_name, full_name=name, sha=pinned[name], license=candidates[name]["license"],
            language=candidates[name]["language"], shortlist_id=shortlist_id, status="queued",
        )
        db.add(env)
        envs.append(env)
    db.commit()
    return envs


def build_repo(db: Session, env_name: str, deps: RepoBuildDeps) -> RepoEnvironment:
    """Snapshot, screen, build, and launch one approved repo, recording the outcome."""
    env = db.get(RepoEnvironment, env_name)
    if env is None:
        raise KeyError(f"no repo environment named {env_name!r}")
    sandbox = db.get(SandboxEnvironment, env_name)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{env_name}-", dir=deps.work_root))
    try:
        _set(db, env, sandbox, "snapshotting")
        root = deps.download(env.full_name, env.sha, work_dir)
        verdict = screen_repo(root, select_files(root), deps.safety)
        if not verdict.accepted:
            reasons = "; ".join(f"{path}: {', '.join(rules)}" for path, rules in sorted(verdict.rejections.items()))
            return _finish(db, env, sandbox, "rejected", reason=reasons)
        _set(db, env, sandbox, "building")
        result = deps.build(env_name, root, env.language)
        env.toolchain, env.test_command = result.toolchain, result.test_command
        env.stable_tests = json.dumps(result.stable_tests)
        if result.status != "ready":
            return _finish(db, env, sandbox, "dropped", reason=result.reason, log=result.log)
        sandbox.container_id = deps.launch(env_name, result.image_tag)
        sandbox.image_tag = result.image_tag
        return _finish(db, env, sandbox, "ready", sandbox_status="running")
    except Exception as exc:
        logger.exception("repo build failed for %s (%s@%s)", env_name, env.full_name, env.sha[:12])
        return _finish(db, env, sandbox, "failed", reason=str(exc)[:2000])
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def fail_build(db: Session, env_name: str, reason: str) -> RepoEnvironment:
    """Record a build that could not start, such as one missing its configuration."""
    env = db.get(RepoEnvironment, env_name)
    if env is None:
        raise KeyError(f"no repo environment named {env_name!r}")
    return _finish(db, env, db.get(SandboxEnvironment, env_name), "failed", reason=reason)


def _live(db: Session, env_name: str) -> bool:
    """Whether the name belongs to a sandbox still in use, as sandbox creation decides it."""
    sandbox = db.get(SandboxEnvironment, env_name)
    return sandbox is not None and sandbox.status not in _STALE_SANDBOX


def _clear_stale(db: Session, env_name: str) -> None:
    """Drop rows left by a deleted or expired build of the same repo."""
    for row in (db.get(SandboxEnvironment, env_name), db.get(RepoEnvironment, env_name)):
        if row is not None:
            db.delete(row)
    db.flush()


def _set(db: Session, env: RepoEnvironment, sandbox: SandboxEnvironment | None, status: str) -> None:
    env.status = status
    if sandbox is not None:
        sandbox.status = "building"
    db.commit()


def _finish(
    db: Session, env: RepoEnvironment, sandbox: SandboxEnvironment | None, status: str,
    *, reason: str = "", log: str = "", sandbox_status: str = "error",
) -> RepoEnvironment:
    env.status, env.reason, env.log = status, reason or None, log or None
    if sandbox is not None:
        sandbox.status = sandbox_status
    db.commit()
    return env


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
