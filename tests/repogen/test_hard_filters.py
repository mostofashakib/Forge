"""Hard filters are enforced in code, whatever query the LLM wrote."""
from datetime import datetime, timedelta, timezone

import pytest

from forge.repogen.discovery import hard_filter, search_qualifiers
from forge.repogen.types import DiscoveryCriteria, RepoCandidate

NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)


def _candidate(**overrides) -> RepoCandidate:
    fields = dict(
        full_name="acme/widgets", default_branch="main", license="MIT", stars=500,
        pushed_at=NOW - timedelta(days=30), archived=False, fork=False, language="Python",
        description="Widgets service", html_url="https://github.com/acme/widgets",
    )
    fields.update(overrides)
    return RepoCandidate(**fields)


def test_a_candidate_meeting_every_rule_passes():
    assert hard_filter(_candidate(), DiscoveryCriteria(), now=NOW) is None


@pytest.mark.parametrize("overrides,reason", [
    ({"license": "GPL-3.0"}, "license GPL-3.0 is not permissive"),
    ({"license": None}, "no license"),
    ({"license": "NOASSERTION"}, "license NOASSERTION is not permissive"),
    ({"archived": True}, "archived"),
    ({"fork": True}, "a fork"),
    ({"stars": 49}, "49 stars, below 50"),
    ({"pushed_at": NOW - timedelta(days=731)}, "no push in 731 days"),
    ({"pushed_at": None}, "no push date"),
    ({"language": "Go"}, "language Go is not supported"),
    ({"language": None}, "language unknown is not supported"),
])
def test_each_rule_rejects_with_a_reason(overrides, reason):
    assert hard_filter(_candidate(**overrides), DiscoveryCriteria(), now=NOW) == reason


def test_criteria_tighten_the_defaults():
    criteria = DiscoveryCriteria(min_stars=1000, max_age_days=90, languages=("TypeScript",))
    assert hard_filter(_candidate(stars=999, language="TypeScript"), criteria, now=NOW) == "999 stars, below 1000"
    assert hard_filter(_candidate(stars=5000), criteria, now=NOW) == "language Python is not supported"


def test_criteria_cannot_add_an_unsupported_language():
    with pytest.raises(ValueError, match="Rust"):
        DiscoveryCriteria(languages=("Python", "Rust"))


def test_criteria_reject_non_positive_limits():
    with pytest.raises(ValueError):
        DiscoveryCriteria(min_stars=-1)
    with pytest.raises(ValueError):
        DiscoveryCriteria(limit=0)


def test_qualifiers_carry_every_hard_filter():
    qualifiers = search_qualifiers(DiscoveryCriteria(min_stars=100, max_age_days=365), now=NOW)
    assert "archived:false" in qualifiers and "fork:false" in qualifiers
    assert "stars:>=100" in qualifiers
    assert "pushed:>=2025-10-11" in qualifiers
    assert "language:python" in qualifiers and "language:typescript" in qualifiers
    for spdx in ("mit", "apache-2.0", "bsd-3-clause", "isc"):
        assert f"license:{spdx}" in qualifiers
