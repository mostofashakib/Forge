"""The repo build task records a build that cannot start instead of leaving it queued."""
from datetime import datetime, timezone

from backend.app.models import RepoEnvironment, SandboxEnvironment


def test_a_missing_token_marks_the_build_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/tasks.db")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    from backend.app import database
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    database.init_db()
    with database.get_session_factory()() as db:
        db.add(SandboxEnvironment(id="repo_acme_widgets", status="queued", env_type="repo",
                                  expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc)))
        db.add(RepoEnvironment(env_name="repo_acme_widgets", full_name="acme/widgets", sha="a" * 40,
                               license="MIT", language="Python", status="queued"))
        db.commit()

    from backend.app.worker.repo_tasks import build_repo_task
    build_repo_task.run("repo_acme_widgets")

    with database.get_session_factory()() as db:
        env = db.get(RepoEnvironment, "repo_acme_widgets")
        assert env.status == "failed" and "GITHUB_TOKEN" in env.reason
        assert db.get(SandboxEnvironment, "repo_acme_widgets").status == "error"
