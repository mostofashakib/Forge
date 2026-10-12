"""Repos API: discover, review, approve, and list repo environments."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app.api import repos as repos_api
from backend.app.main import app
from forge.repogen.discovery import RepoFit, RepoRanking, SearchPlan
from forge.repogen.github_mcp import GitHubMcpError
from forge.repogen.types import RepoCandidate
from forge.safety.pseudonymizer import Pseudonymizer
from forge.safety.service import DatasetSafetyService, PiiScrubbing


def _repo(name, **overrides):
    fields = dict(
        full_name=name, default_branch="main", license="MIT", stars=900,
        pushed_at=datetime.now(timezone.utc) - timedelta(days=5), archived=False, fork=False,
        language="Python", description="d", html_url=f"https://github.com/{name}",
    )
    fields.update(overrides)
    return RepoCandidate(**fields)


class FakeSource:
    def __init__(self, results, fail=None):
        self.results = results
        self.fail = fail

    def search(self, query, limit):
        if self.fail:
            raise self.fail
        return self.results

    def root_listing(self, full_name, ref):
        return ["README.md", "tests/"]

    def file_text(self, full_name, path, ref):
        return "# readme"

    def head_sha(self, full_name, branch):
        return "a" * 40


class FakeLLM:
    def extract(self, system, user, schema):
        if schema is SearchPlan:
            return SearchPlan(queries=["http clients"])
        return RepoRanking(repos=[RepoFit(full_name="acme/widgets", fit=0.9, has_tests=True, reason="fits")])


class NoNames:
    def detect_many(self, texts):
        return [[] for _ in texts]


class NoAdult:
    def detect(self, texts):
        return []


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_URL", f"sqlite:///{tmp_path}/repos.db")
    from backend.app import database
    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    database.init_db()
    dispatched = []
    source = FakeSource([_repo("acme/widgets"), _repo("x/gpl", license="GPL-3.0")])
    app.dependency_overrides[repos_api.get_github_source] = lambda: source
    app.dependency_overrides[repos_api.get_discovery_llm] = lambda: FakeLLM()
    app.dependency_overrides[repos_api.get_readme_safety] = lambda: DatasetSafetyService(
        NoAdult(), PiiScrubbing(Pseudonymizer(b"k"), NoNames())
    )
    app.dependency_overrides[repos_api.get_build_dispatcher] = lambda: dispatched.append
    test_client = TestClient(app)
    test_client.dispatched = dispatched
    test_client.source = source
    yield test_client
    app.dependency_overrides.clear()


def _discover(client, **body):
    return client.post("/api/repos/discover", json={"brief": "Python HTTP clients with tests", **body})


def test_discover_returns_a_saved_shortlist(client):
    res = _discover(client)
    assert res.status_code == 200
    body = res.json()
    assert [c["full_name"] for c in body["candidates"]] == ["acme/widgets"]
    assert body["rejected"] == {"x/gpl": "license GPL-3.0 is not permissive"}
    assert client.get(f"/api/repos/shortlists/{body['id']}").json() == body


def test_discover_rejects_bad_criteria(client):
    assert _discover(client, languages=["Rust"]).status_code == 422
    assert _discover(client, limit=0).status_code == 422
    assert client.post("/api/repos/discover", json={"brief": ""}).status_code == 422


def test_a_github_failure_is_a_bad_gateway(client):
    client.source.fail = GitHubMcpError("search_repositories failed: rate limit")
    res = _discover(client)
    assert res.status_code == 502 and "rate limit" in res.json()["detail"]


def test_approve_queues_builds_and_lists_them(client):
    shortlist_id = _discover(client).json()["id"]
    res = client.post(f"/api/repos/shortlists/{shortlist_id}/approve", json={"full_names": ["acme/widgets"]})

    assert res.status_code == 202
    assert [env["env_name"] for env in res.json()["environments"]] == ["repo_acme_widgets"]
    assert client.dispatched == ["repo_acme_widgets"]
    (listed,) = client.get("/api/repos").json()
    assert (listed["full_name"], listed["status"], listed["sha"]) == ("acme/widgets", "queued", "a" * 40)


def test_approve_errors_map_to_clear_statuses(client):
    shortlist_id = _discover(client).json()["id"]
    approve = lambda sid, names: client.post(f"/api/repos/shortlists/{sid}/approve", json={"full_names": names})
    assert approve("rs_missing", ["acme/widgets"]).status_code == 404
    assert approve(shortlist_id, ["x/gpl"]).status_code == 422
    assert approve(shortlist_id, []).status_code == 422
    assert approve(shortlist_id, ["acme/widgets"]).status_code == 202
    assert approve(shortlist_id, ["acme/widgets"]).status_code == 409


def test_approval_past_the_environment_cap_is_refused(client, monkeypatch):
    monkeypatch.setenv("FORGE_SANDBOX_LIMIT", "0")
    shortlist_id = _discover(client).json()["id"]
    res = client.post(f"/api/repos/shortlists/{shortlist_id}/approve", json={"full_names": ["acme/widgets"]})
    assert res.status_code == 429
    assert client.dispatched == []


def test_a_missing_github_token_is_a_clear_503(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(Exception) as info:
        next(repos_api.get_github_source())
    assert info.value.status_code == 503 and "GITHUB_TOKEN" in info.value.detail
