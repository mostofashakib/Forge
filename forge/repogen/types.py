"""Values passed between repo discovery, snapshotting, and building."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

SUPPORTED_LANGUAGES = ("Python", "TypeScript", "JavaScript")


@dataclass(frozen=True)
class RepoCandidate:
    """One repository as GitHub search describes it."""

    full_name: str
    default_branch: str
    license: str | None
    stars: int
    pushed_at: datetime | None
    archived: bool
    fork: bool
    language: str | None
    description: str
    html_url: str


@dataclass(frozen=True)
class DiscoveryCriteria:
    """Hard filters a person sets. Code enforces them, not the LLM."""

    min_stars: int = 50
    max_age_days: int = 730
    languages: tuple[str, ...] = SUPPORTED_LANGUAGES
    limit: int = 10

    def __post_init__(self) -> None:
        unsupported = [language for language in self.languages if language not in SUPPORTED_LANGUAGES]
        if unsupported or not self.languages:
            raise ValueError(f"languages must be a non-empty subset of {SUPPORTED_LANGUAGES}, got {unsupported}")
        if self.min_stars < 0 or self.max_age_days < 1 or self.limit < 1:
            raise ValueError("min_stars must be >= 0, and max_age_days and limit must be >= 1")


@dataclass(frozen=True)
class RankedRepo:
    candidate: RepoCandidate
    fit: float
    reason: str


@dataclass(frozen=True)
class Shortlist:
    brief: str
    repos: list[RankedRepo]
    # Candidates the filters or the ranker turned away, with why.
    rejected: dict[str, str] = field(default_factory=dict)
