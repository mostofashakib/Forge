"""Repos API: turn a brief into a shortlist, approve repos, and track their builds."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import RepoEnvironment, RepoShortlist
from backend.app.services import repo_pipeline
from forge.extraction.llm_client import LLMClient, get_client
from forge.repogen.discovery import DiscoveryError, discover
from forge.repogen.github_mcp import (
    GitHubMcpError,
    GitHubSource,
    McpGitHubSource,
    RepoConfigError,
    github_mcp_session,
    github_settings_from_env,
)
from forge.repogen.types import SUPPORTED_LANGUAGES, DiscoveryCriteria
from forge.safety.config import SafetyConfigError, safety_service_from_env
from forge.safety.service import DatasetSafetyService
from forge.safety.types import DataSource
from forge.settings import sandbox_limit

router = APIRouter(prefix="/api/repos")


class DiscoverRequest(BaseModel):
    brief: str = Field(min_length=3, max_length=2000)
    min_stars: int = 50
    max_age_days: int = 730
    languages: list[str] = Field(default_factory=lambda: list(SUPPORTED_LANGUAGES))
    limit: int = Field(default=10, ge=1, le=25)


class ApproveRequest(BaseModel):
    full_names: list[str] = Field(min_length=1, max_length=25)


def get_github_source() -> Iterator[GitHubSource]:
    try:
        url, token = github_settings_from_env()
    except RepoConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    with ExitStack() as stack:
        try:
            call = stack.enter_context(github_mcp_session(url, token))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"could not reach the GitHub MCP server: {exc}") from exc
        yield McpGitHubSource(call)


def get_discovery_llm() -> LLMClient:
    return get_client()


def get_readme_safety() -> DatasetSafetyService:
    try:
        return safety_service_from_env(DataSource.INTERNET).service
    except SafetyConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_build_dispatcher() -> Callable[[str], None]:
    from backend.app.worker.repo_tasks import build_repo_task

    return lambda env_name: build_repo_task.delay(env_name=env_name)


@router.post("/discover")
def discover_repos(
    body: DiscoverRequest,
    db: Session = Depends(get_db),
    source: GitHubSource = Depends(get_github_source),
    llm: LLMClient = Depends(get_discovery_llm),
    safety: DatasetSafetyService = Depends(get_readme_safety),
) -> dict:
    try:
        criteria = DiscoveryCriteria(
            min_stars=body.min_stars, max_age_days=body.max_age_days,
            languages=tuple(body.languages), limit=body.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        shortlist = discover(body.brief, criteria, source, llm, safety, now=datetime.now(timezone.utc))
    except GitHubMcpError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except DiscoveryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return repo_pipeline.shortlist_view(repo_pipeline.save_shortlist(db, shortlist, criteria))


@router.get("/shortlists/{shortlist_id}")
def get_shortlist(shortlist_id: str, db: Session = Depends(get_db)) -> dict:
    row = db.get(RepoShortlist, shortlist_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Shortlist not found")
    return repo_pipeline.shortlist_view(row)


@router.post("/shortlists/{shortlist_id}/approve", status_code=202)
def approve_repos(
    shortlist_id: str,
    body: ApproveRequest,
    db: Session = Depends(get_db),
    source: GitHubSource = Depends(get_github_source),
    dispatch: Callable[[str], None] = Depends(get_build_dispatcher),
) -> dict:
    try:
        envs = repo_pipeline.approve(db, shortlist_id, body.full_names, source, limit=sandbox_limit())
    except repo_pipeline.ShortlistNotFound as exc:
        raise HTTPException(status_code=404, detail="Shortlist not found") from exc
    except repo_pipeline.NotInShortlist as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except repo_pipeline.AlreadyApproved as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except repo_pipeline.CapacityExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except GitHubMcpError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    for env in envs:
        dispatch(env.env_name)
    return {"environments": [_env_view(env) for env in envs]}


@router.get("")
def list_repo_environments(db: Session = Depends(get_db)) -> list[dict]:
    return [_env_view(env) for env in db.query(RepoEnvironment).order_by(RepoEnvironment.created_at).all()]


def _env_view(env: RepoEnvironment) -> dict:
    return {
        "env_name": env.env_name,
        "full_name": env.full_name,
        "sha": env.sha,
        "license": env.license,
        "language": env.language,
        "status": env.status,
        "reason": env.reason,
        "toolchain": env.toolchain,
        "stable_test_count": len(json.loads(env.stable_tests or "[]")),
    }
