"""Find repos for a plain-English brief: LLM-planned searches, code-enforced filters.

The LLM writes the free-text part of each search and ranks what comes back.
It never decides eligibility: every result is re-checked against the hard
filters in code, so a query that ignores the qualifiers changes nothing.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from forge.repogen.types import DiscoveryCriteria, RepoCandidate
from forge.safety.copyright import PERMISSIVE_LICENSES


def hard_filter(candidate: RepoCandidate, criteria: DiscoveryCriteria, *, now: datetime) -> str | None:
    """Why `candidate` is ineligible, or None when it passes every rule."""
    if candidate.license is None:
        return "no license"
    if candidate.license not in PERMISSIVE_LICENSES:
        return f"license {candidate.license} is not permissive"
    if candidate.archived:
        return "archived"
    if candidate.fork:
        return "a fork"
    if candidate.stars < criteria.min_stars:
        return f"{candidate.stars} stars, below {criteria.min_stars}"
    if candidate.pushed_at is None:
        return "no push date"
    age_days = (now - candidate.pushed_at).days
    if age_days > criteria.max_age_days:
        return f"no push in {age_days} days"
    if candidate.language not in criteria.languages:
        return f"language {candidate.language or 'unknown'} is not supported"
    return None


def search_qualifiers(criteria: DiscoveryCriteria, *, now: datetime) -> str:
    """GitHub search qualifiers that mirror the hard filters, to keep results relevant."""
    pushed_since = (now - timedelta(days=criteria.max_age_days)).date().isoformat()
    parts = ["archived:false", "fork:false", f"stars:>={criteria.min_stars}", f"pushed:>={pushed_since}"]
    parts += [f"language:{language.lower()}" for language in criteria.languages]
    parts += [f"license:{spdx.lower()}" for spdx in sorted(PERMISSIVE_LICENSES)]
    return " ".join(parts)
