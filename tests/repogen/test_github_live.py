"""Live check against GitHub's MCP server. Runs only when GITHUB_TOKEN is set."""
import os

import pytest

from forge.repogen.github_mcp import McpGitHubSource, github_mcp_session, github_settings_from_env

pytestmark = pytest.mark.skipif(not os.environ.get("GITHUB_TOKEN"), reason="GITHUB_TOKEN not set")


def test_search_listing_and_head_sha_against_real_github():
    url, token = github_settings_from_env()
    with github_mcp_session(url, token) as call:
        source = McpGitHubSource(call)
        results = source.search("http client license:mit language:python stars:>=1000", limit=3)
        assert results and all(repo.license == "MIT" for repo in results)
        repo = results[0]
        assert any(entry.lower().startswith("readme") for entry in source.root_listing(repo.full_name, repo.default_branch))
        assert len(source.head_sha(repo.full_name, repo.default_branch)) == 40
