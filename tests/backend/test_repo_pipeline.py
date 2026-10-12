"""Repo pipeline: shortlists, approval within the sandbox cap, and per-repo builds."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import RepoEnvironment, RepoShortlist, SandboxEnvironment
from backend.app.services import repo_pipeline
from backend.app.services.repo_pipeline import (
    AlreadyApproved,
    CapacityExceeded,
    NotInShortlist,
    RepoBuildDeps,
    ShortlistNotFound,
    approve,
    build_repo,
    repo_env_name,
    fail_build,
    save_shortlist,
)
from forge.repogen.build import RepoBuildResult
from forge.repogen.types import DiscoveryCriteria, RankedRepo, RepoCandidate, Shortlist
from forge.safety.pseudonymizer import Pseudonymizer
from forge.safety.service import DatasetSafetyService, PiiScrubbing


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _candidate(name, language="Python"):
    return RepoCandidate(
        full_name=name, default_branch="main", license="MIT", stars=900,
        pushed_at=datetime(2026, 9, 1, tzinfo=timezone.utc), archived=False, fork=False,
        language=language, description="d", html_url=f"https://github.com/{name}",
    )


def _shortlist(db, *names):
    shortlist = Shortlist("brief", [RankedRepo(_candidate(n), 0.9, "fits") for n in names], {"x/gpl": "license GPL-3.0 is not permissive"})
    return save_shortlist(db, shortlist, DiscoveryCriteria())


class FakeSource:
    def head_sha(self, full_name, branch):
        return (full_name.replace("/", "") * 40)[:40].lower().replace("_", "a")


def test_env_names_are_valid_sandbox_ids():
    assert repo_env_name("Acme/My.Widgets") == "repo_acme_my_widgets"


def test_a_saved_shortlist_round_trips(db):
    row = _shortlist(db, "acme/widgets")
    view = repo_pipeline.shortlist_view(row)
    assert view["id"] == row.id and view["brief"] == "brief"
    assert [c["full_name"] for c in view["candidates"]] == ["acme/widgets"]
    assert view["rejected"] == {"x/gpl": "license GPL-3.0 is not permissive"}


def test_approval_pins_shas_and_queues_repo_sandboxes(db):
    row = _shortlist(db, "acme/widgets", "acme/gadgets")
    envs = approve(db, row.id, ["acme/widgets"], FakeSource(), limit=10)

    (env,) = envs
    assert env.env_name == "repo_acme_widgets" and env.status == "queued" and len(env.sha) == 40
    assert (env.license, env.language, env.shortlist_id) == ("MIT", "Python", row.id)
    sandbox = db.get(SandboxEnvironment, "repo_acme_widgets")
    assert (sandbox.env_type, sandbox.status) == ("repo", "queued")


def test_approving_an_unknown_shortlist_fails(db):
    with pytest.raises(ShortlistNotFound):
        approve(db, "rs_missing", ["acme/widgets"], FakeSource(), limit=10)


def test_approving_a_repo_not_in_the_shortlist_fails_and_stores_nothing(db):
    row = _shortlist(db, "acme/widgets")
    with pytest.raises(NotInShortlist, match="x/gpl"):
        approve(db, row.id, ["acme/widgets", "x/gpl"], FakeSource(), limit=10)
    assert db.query(RepoEnvironment).count() == 0


def test_approval_respects_the_sandbox_cap(db):
    for i in range(9):
        db.add(SandboxEnvironment(id=f"env{i}", status="running", env_type="cli",
                                  expires_at=datetime.now(timezone.utc) + timedelta(days=1)))
    db.commit()
    row = _shortlist(db, "acme/widgets", "acme/gadgets")
    with pytest.raises(CapacityExceeded, match="10"):
        approve(db, row.id, ["acme/widgets", "acme/gadgets"], FakeSource(), limit=10)
    assert db.query(RepoEnvironment).count() == 0


def test_approving_a_repo_twice_is_refused(db):
    row = _shortlist(db, "acme/widgets")
    approve(db, row.id, ["acme/widgets"], FakeSource(), limit=10)
    with pytest.raises(AlreadyApproved, match="acme/widgets"):
        approve(db, row.id, ["acme/widgets"], FakeSource(), limit=10)


# --- building ---------------------------------------------------------------

class NoNames:
    def detect_many(self, texts):
        return [[] for _ in texts]


class NoAdult:
    def detect(self, texts):
        return []


def _queued(db, name="acme/widgets"):
    row = _shortlist(db, name)
    (env,) = approve(db, row.id, [name], FakeSource(), limit=10)
    return env.env_name


def _deps(tmp_path, files, result=None, download_error=None):
    calls = {"launch": []}

    def download(full_name, sha, dest_dir):
        if download_error:
            raise download_error
        root = dest_dir / "src"
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        return root

    def build(env_name, root, language):
        calls["built_root"] = root
        calls["files"] = {path.name: path.read_text() for path in root.rglob("*") if path.is_file()}
        return result or RepoBuildResult("ready", image_tag="forge-repo-x:latest", toolchain="python",
                                         test_command="python -m pytest", stable_tests=["t::a"])

    def launch(env_name, image):
        calls["launch"].append((env_name, image))
        return "container-1"

    safety = DatasetSafetyService(NoAdult(), PiiScrubbing(Pseudonymizer(b"k"), NoNames()))
    return RepoBuildDeps(download=download, safety=safety, build=build, launch=launch, work_root=tmp_path), calls


def test_a_clean_repo_builds_launches_and_becomes_a_ready_sandbox(db, tmp_path):
    env_name = _queued(db)
    deps, calls = _deps(tmp_path, {"app.py": "x = 'jane@realcorp.com'\n"})

    env = build_repo(db, env_name, deps)

    assert (env.status, env.toolchain, json.loads(env.stable_tests)) == ("ready", "python", ["t::a"])
    sandbox = db.get(SandboxEnvironment, env_name)
    assert (sandbox.status, sandbox.container_id, sandbox.image_tag) == ("running", "container-1", "forge-repo-x:latest")
    assert calls["launch"] == [(env_name, "forge-repo-x:latest")]
    assert "realcorp" not in calls["files"]["app.py"]


def test_a_flagged_file_rejects_the_repo_without_building(db, tmp_path):
    env_name = _queued(db)
    deps, calls = _deps(tmp_path, {"lib.c": "/* SPDX-License-Identifier: GPL-3.0-only */\n"})
    env = build_repo(db, env_name, deps)
    assert env.status == "rejected" and "lib.c" in env.reason and "copyright" in env.reason
    assert "built_root" not in calls and calls["launch"] == []
    assert db.get(SandboxEnvironment, env_name).status == "error"


def test_a_dropped_build_records_why_and_its_log(db, tmp_path):
    env_name = _queued(db)
    deps, _ = _deps(tmp_path, {"app.py": "x = 1\n"},
                    result=RepoBuildResult("dropped", "install failed", toolchain="python", log="pip exploded"))
    env = build_repo(db, env_name, deps)
    assert (env.status, env.reason, env.log) == ("dropped", "install failed", "pip exploded")
    assert db.get(SandboxEnvironment, env_name).status == "error"


def test_an_unexpected_failure_marks_the_repo_failed(db, tmp_path):
    env_name = _queued(db)
    deps, _ = _deps(tmp_path, {}, download_error=RuntimeError("HTTP 502 from GitHub"))
    env = build_repo(db, env_name, deps)
    assert env.status == "failed" and "HTTP 502" in env.reason


def test_building_an_unknown_repo_env_fails_loudly(db, tmp_path):
    deps, _ = _deps(tmp_path, {})
    with pytest.raises(KeyError):
        build_repo(db, "repo_missing", deps)


def test_the_work_directory_is_removed_afterwards(db, tmp_path):
    env_name = _queued(db)
    deps, calls = _deps(tmp_path, {"app.py": "x = 1\n"})
    build_repo(db, env_name, deps)
    assert not calls["built_root"].exists()


def test_a_repo_can_be_approved_again_after_its_sandbox_is_deleted(db):
    row = _shortlist(db, "acme/widgets")
    approve(db, row.id, ["acme/widgets"], FakeSource(), limit=10)
    db.delete(db.get(SandboxEnvironment, "repo_acme_widgets"))
    db.commit()

    (env,) = approve(db, row.id, ["acme/widgets"], FakeSource(), limit=10)
    assert env.status == "queued"
    assert db.query(RepoEnvironment).count() == 1


def test_a_repo_whose_sandbox_expired_can_be_approved_again(db):
    row = _shortlist(db, "acme/widgets")
    approve(db, row.id, ["acme/widgets"], FakeSource(), limit=10)
    db.get(SandboxEnvironment, "repo_acme_widgets").status = "expired"
    db.commit()
    (env,) = approve(db, row.id, ["acme/widgets"], FakeSource(), limit=10)
    assert db.get(SandboxEnvironment, env.env_name).status == "queued"


def test_a_build_that_cannot_start_is_marked_failed(db):
    env_name = _queued(db)
    env = fail_build(db, env_name, "GITHUB_TOKEN is not set")
    assert (env.status, env.reason) == ("failed", "GITHUB_TOKEN is not set")
    assert db.get(SandboxEnvironment, env_name).status == "error"
