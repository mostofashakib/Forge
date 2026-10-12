"""GitHub access through GitHub's MCP server.

`McpGitHubSource` turns MCP tool results into plain values. It takes any
callable that runs one tool, so tests pass recorded payloads and production
passes a live session from `github_mcp_session`.

Search asks for full repository objects (`minimal_output=false`) because the
minimal form leaves out the license, which eligibility depends on.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Any, Protocol

from mcp.types import CallToolResult, EmbeddedResource, TextContent

from forge.repogen.types import RepoCandidate

DEFAULT_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
_MAX_PER_PAGE = 100
_TIMEOUT_S = 60.0


class RepoConfigError(RuntimeError):
    """Repo discovery cannot run, with the variable to fix in the message."""


class GitHubMcpError(RuntimeError):
    """The GitHub MCP server refused a call or answered in an unexpected shape."""


class ToolCaller(Protocol):
    def __call__(self, name: str, arguments: dict[str, Any]) -> CallToolResult: ...


class GitHubSource(Protocol):
    def search(self, query: str, limit: int) -> list[RepoCandidate]: ...

    def root_listing(self, full_name: str, ref: str) -> list[str]: ...

    def file_text(self, full_name: str, path: str, ref: str) -> str: ...

    def head_sha(self, full_name: str, branch: str) -> str: ...


class McpGitHubSource:
    def __init__(self, call: ToolCaller) -> None:
        self._call = call

    def search(self, query: str, limit: int) -> list[RepoCandidate]:
        payload = self._json("search_repositories", {
            "query": query, "perPage": min(limit, _MAX_PER_PAGE), "minimal_output": False,
        })
        try:
            return [_candidate(item) for item in payload["items"]]
        except (KeyError, TypeError) as exc:
            raise GitHubMcpError(f"search_repositories returned an unexpected shape: {exc}") from exc

    def root_listing(self, full_name: str, ref: str) -> list[str]:
        owner, repo = _split(full_name)
        entries = self._json("get_file_contents", {"owner": owner, "repo": repo, "path": "/", "ref": ref})
        try:
            return [entry["name"] + ("/" if entry.get("type") == "dir" else "") for entry in entries]
        except (KeyError, TypeError, AttributeError) as exc:
            raise GitHubMcpError(f"get_file_contents returned an unexpected listing: {exc}") from exc

    def file_text(self, full_name: str, path: str, ref: str) -> str:
        owner, repo = _split(full_name)
        result = self._checked("get_file_contents", {"owner": owner, "repo": repo, "path": path, "ref": ref})
        embedded = [
            item.resource.text for item in result.content
            if isinstance(item, EmbeddedResource) and hasattr(item.resource, "text")
        ]
        return embedded[0] if embedded else "\n".join(_texts(result))

    def head_sha(self, full_name: str, branch: str) -> str:
        owner, repo = _split(full_name)
        commits = self._json("list_commits", {"owner": owner, "repo": repo, "sha": branch, "perPage": 1})
        if not commits:
            raise GitHubMcpError(f"{full_name}@{branch} has no commits")
        return commits[0]["sha"]

    def _checked(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        result = self._call(name, arguments)
        if result.isError:
            raise GitHubMcpError(f"{name} failed: {' '.join(_texts(result))[:500]}")
        return result

    def _json(self, name: str, arguments: dict[str, Any]) -> Any:
        text = "".join(_texts(self._checked(name, arguments)))
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GitHubMcpError(f"{name} did not return JSON: {text[:200]!r}") from exc


def github_settings_from_env() -> tuple[str, str]:
    """The MCP endpoint and token. A missing token raises naming `GITHUB_TOKEN`."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RepoConfigError("GITHUB_TOKEN is not set. Repo discovery needs a GitHub token with public read access.")
    return os.environ.get("FORGE_GITHUB_MCP_URL") or DEFAULT_GITHUB_MCP_URL, token


@contextmanager
def github_mcp_session(url: str, token: str) -> Iterator[ToolCaller]:
    """A live MCP session, callable from sync code, open for the block's duration."""
    from anyio.from_thread import start_blocking_portal

    with start_blocking_portal() as portal:
        with portal.wrap_async_context_manager(_open_session(url, token)) as session:
            yield lambda name, arguments: portal.call(session.call_tool, name, arguments)


@asynccontextmanager
async def _open_session(url: str, token: str):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(headers=headers, timeout=_TIMEOUT_S) as http:
        async with streamable_http_client(url, http_client=http) as (read, write, _session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def _texts(result: CallToolResult) -> list[str]:
    return [item.text for item in result.content if isinstance(item, TextContent)]


def _split(full_name: str) -> tuple[str, str]:
    owner, _, repo = full_name.partition("/")
    if not owner or not repo or "/" in repo:
        raise ValueError(f"expected owner/repo, got {full_name!r}")
    return owner, repo


def _candidate(item: dict[str, Any]) -> RepoCandidate:
    license_info = item.get("license") or {}
    pushed = item.get("pushed_at")
    return RepoCandidate(
        full_name=item["full_name"],
        default_branch=item.get("default_branch") or "main",
        license=license_info.get("spdx_id"),
        stars=int(item.get("stargazers_count") or 0),
        pushed_at=datetime.fromisoformat(pushed) if pushed else None,
        archived=bool(item.get("archived")),
        fork=bool(item.get("fork")),
        language=item.get("language"),
        description=item.get("description") or "",
        html_url=item.get("html_url") or "",
    )
