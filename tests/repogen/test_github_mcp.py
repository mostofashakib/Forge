"""The MCP-backed GitHub source parses the official server's payloads."""
import json
from datetime import datetime, timezone

import pytest
from mcp.types import CallToolResult, EmbeddedResource, TextContent, TextResourceContents

from forge.repogen.github_mcp import (
    DEFAULT_GITHUB_MCP_URL,
    GitHubMcpError,
    McpGitHubSource,
    RepoConfigError,
    github_settings_from_env,
)

SEARCH_PAYLOAD = {
    "total_count": 2,
    "incomplete_results": False,
    "items": [
        {
            "full_name": "acme/widgets", "default_branch": "main", "stargazers_count": 812,
            "pushed_at": "2026-09-01T12:00:00Z", "archived": False, "fork": False,
            "language": "Python", "description": "Widgets API", "html_url": "https://github.com/acme/widgets",
            "license": {"key": "mit", "spdx_id": "MIT"},
        },
        {
            "full_name": "acme/legacy", "default_branch": "master", "stargazers_count": 3,
            "pushed_at": None, "archived": True, "fork": True, "language": None,
            "description": None, "html_url": "https://github.com/acme/legacy", "license": None,
        },
    ],
}


def _text(payload) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])


class FakeCaller:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, name, arguments):
        self.calls.append((name, arguments))
        return self.responses[name]


def test_search_parses_full_repository_objects():
    caller = FakeCaller({"search_repositories": _text(SEARCH_PAYLOAD)})
    widgets, legacy = McpGitHubSource(caller).search("widgets api", limit=5)

    assert caller.calls == [("search_repositories", {"query": "widgets api", "perPage": 5, "minimal_output": False})]
    assert widgets.full_name == "acme/widgets" and widgets.license == "MIT" and widgets.stars == 812
    assert widgets.pushed_at == datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    assert legacy.license is None and legacy.pushed_at is None and legacy.archived and legacy.fork
    assert legacy.description == ""


def test_search_caps_page_size_at_the_api_limit():
    caller = FakeCaller({"search_repositories": _text({"items": []})})
    assert McpGitHubSource(caller).search("q", limit=500) == []
    assert caller.calls[0][1]["perPage"] == 100


def test_root_listing_marks_directories():
    caller = FakeCaller({"get_file_contents": _text([
        {"type": "file", "name": "README.md", "path": "README.md"},
        {"type": "dir", "name": "tests", "path": "tests"},
    ])})
    assert McpGitHubSource(caller).root_listing("acme/widgets", "main") == ["README.md", "tests/"]
    assert caller.calls == [("get_file_contents", {"owner": "acme", "repo": "widgets", "path": "/", "ref": "main"})]


def test_file_text_reads_the_embedded_resource():
    result = CallToolResult(content=[
        TextContent(type="text", text="successfully downloaded text file (SHA: abc)"),
        EmbeddedResource(type="resource", resource=TextResourceContents(uri="repo://acme/widgets/README.md", text="# Widgets")),
    ])
    caller = FakeCaller({"get_file_contents": result})
    assert McpGitHubSource(caller).file_text("acme/widgets", "README.md", "main") == "# Widgets"


def test_head_sha_reads_the_newest_commit():
    caller = FakeCaller({"list_commits": _text([{"sha": "deadbeef" * 5, "commit": {"message": "x"}}])})
    assert McpGitHubSource(caller).head_sha("acme/widgets", "main") == "deadbeef" * 5
    assert caller.calls == [("list_commits", {"owner": "acme", "repo": "widgets", "sha": "main", "perPage": 1})]


def test_a_branch_without_commits_is_an_error():
    caller = FakeCaller({"list_commits": _text([])})
    with pytest.raises(GitHubMcpError, match="no commits"):
        McpGitHubSource(caller).head_sha("acme/widgets", "main")


def test_a_tool_error_raises_with_the_server_message():
    caller = FakeCaller({"search_repositories": CallToolResult(
        content=[TextContent(type="text", text="API rate limit exceeded")], isError=True
    )})
    with pytest.raises(GitHubMcpError, match="rate limit"):
        McpGitHubSource(caller).search("q", limit=5)


def test_a_malformed_payload_raises_instead_of_returning_nothing():
    caller = FakeCaller({"search_repositories": CallToolResult(content=[TextContent(type="text", text="not json")])})
    with pytest.raises(GitHubMcpError, match="search_repositories"):
        McpGitHubSource(caller).search("q", limit=5)


def test_a_full_name_without_an_owner_is_rejected():
    with pytest.raises(ValueError):
        McpGitHubSource(FakeCaller({})).root_listing("widgets", "main")


def test_settings_need_a_token_and_default_the_url(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("FORGE_GITHUB_MCP_URL", raising=False)
    with pytest.raises(RepoConfigError, match="GITHUB_TOKEN"):
        github_settings_from_env()
    monkeypatch.setenv("GITHUB_TOKEN", "  tok  ")
    assert github_settings_from_env() == (DEFAULT_GITHUB_MCP_URL, "tok")
    monkeypatch.setenv("FORGE_GITHUB_MCP_URL", "http://localhost:8082/mcp")
    assert github_settings_from_env()[0] == "http://localhost:8082/mcp"


def test_the_placeholder_token_is_rejected(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "mock_key")
    with pytest.raises(RepoConfigError, match="GITHUB_TOKEN"):
        github_settings_from_env()


def test_a_listing_in_an_unexpected_shape_raises():
    caller = FakeCaller({"get_file_contents": _text({"type": "file", "name": "README.md"})})
    with pytest.raises(GitHubMcpError, match="get_file_contents"):
        McpGitHubSource(caller).root_listing("acme/widgets", "main")
