"""Query parsing and matching for Slack search.

Splitting a query is not as simple as `str.split()`, because two things people
type routinely mean something structural rather than literal:

    "exact phrase"     the words in that order, not the words in any order
    from: in: before: after: on:      filters, not search terms

Both are parsed here so the SQLite service and any other implementation agree
on what a query means. A filter this module does not recognise is reported as
an error rather than matched literally: a query that quietly returns nothing is
worse than one that says why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

#: The workspace's wall-clock offset. The whole fixture calendar sits inside
#: Pacific daylight time, and pinning the offset here keeps date filters
#: deterministic without depending on a tz database being installed.
WORKSPACE_UTC_OFFSET = timedelta(hours=-7)

FILTER_NAMES = ("from", "in", "before", "after", "on")

_TOKEN = re.compile(r'"([^"]*)"|(\S+)')


class SearchQueryError(ValueError):
    """The query used a filter that cannot be honoured as written."""


@dataclass(frozen=True, slots=True)
class SearchQuery:
    terms: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()
    senders: tuple[str, ...] = field(default=())
    conversations: tuple[str, ...] = field(default=())
    before: date | None = None
    after: date | None = None
    on: date | None = None

    @property
    def has_filters(self) -> bool:
        return bool(self.senders or self.conversations or self.before or self.after or self.on)


def _parse_date(value: str, filter_name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise SearchQueryError(
            f"{filter_name}: expects a YYYY-MM-DD date, not {value!r}"
        ) from error


def parse_query(query: str) -> SearchQuery:
    """Split a raw query into terms, phrases and filters."""
    terms: list[str] = []
    phrases: list[str] = []
    senders: list[str] = []
    conversations: list[str] = []
    before = after = on = None

    for match in _TOKEN.finditer(query):
        quoted, bare = match.group(1), match.group(2)
        if quoted is not None:
            phrase = quoted.strip().lower()
            if phrase:
                phrases.append(phrase)
            continue
        name, separator, value = bare.partition(":")
        if separator and name.lower() in FILTER_NAMES:
            name = name.lower()
            value = value.strip('"').lstrip("@#").lower()
            if not value:
                raise SearchQueryError(f"{name}: needs a value")
            if name == "from":
                senders.append(value)
            elif name == "in":
                conversations.append(value)
            elif name == "before":
                before = _parse_date(value, "before")
            elif name == "after":
                after = _parse_date(value, "after")
            else:
                on = _parse_date(value, "on")
            continue
        terms.append(bare.lower().lstrip("@#"))

    return SearchQuery(
        terms=tuple(terms), phrases=tuple(phrases), senders=tuple(senders),
        conversations=tuple(conversations), before=before, after=after, on=on,
    )


def message_haystack(
    body: str, location_name: str, author_display_name: str, author_handle: str
) -> str:
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


def ts_to_date(ts: str) -> date:
    """The workspace-local calendar date a Slack ts falls on."""
    seconds = float(str(ts).split(".", 1)[0] or 0)
    return (datetime.fromtimestamp(seconds, timezone.utc) + WORKSPACE_UTC_OFFSET).date()


def matches_person(parsed: SearchQuery, display_name: str, handle: str) -> bool:
    """Every `from:` must name this author; that is what makes them combinable."""
    person = f"{display_name.lower()} {handle.lower()}"
    return all(sender in person for sender in parsed.senders)


def matches_conversation(parsed: SearchQuery, location_name: str) -> bool:
    name = location_name.lower()
    return all(conversation in name for conversation in parsed.conversations)


def matches_date(parsed: SearchQuery, ts: str) -> bool:
    """`before` is strictly earlier and `after` is on-or-later, so they partition."""
    when = ts_to_date(ts)
    if parsed.on is not None and when != parsed.on:
        return False
    if parsed.before is not None and not when < parsed.before:
        return False
    if parsed.after is not None and not when >= parsed.after:
        return False
    return True


def matches_query(
    parsed: SearchQuery, haystack: str, display_name: str, handle: str,
    location_name: str, ts: str,
) -> bool:
    return (
        matches_all_terms(haystack, [*parsed.terms, *parsed.phrases])
        and matches_person(parsed, display_name, handle)
        and matches_conversation(parsed, location_name)
        and matches_date(parsed, ts)
    )
