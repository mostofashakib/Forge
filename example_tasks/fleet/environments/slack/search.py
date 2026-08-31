"""Message search rules shared by the in-memory and SQLite Slack services.

Both implementations must rank and match identically, so the term
normalization and haystack construction live here and nowhere else.
"""

from __future__ import annotations


def search_terms(query: str) -> list[str]:
    return [term.lower().lstrip("@#") for term in query.split()]


def message_haystack(body: str, location_name: str, author_display_name: str, author_handle: str) -> str:
    return " ".join(
        [
            body.lower(),
            location_name.lower(),
            author_display_name.lower(),
            author_handle.lower(),
        ]
    )


def matches_all_terms(haystack: str, terms: list[str]) -> bool:
    return all(term in haystack for term in terms)
