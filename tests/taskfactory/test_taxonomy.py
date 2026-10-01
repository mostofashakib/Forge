"""The taxonomy step: one LLM call, a code-side spread check, one retry."""
from __future__ import annotations

import pytest

from forge.taskfactory.profile import EnvironmentProfile, ToolInfo
from forge.taskfactory.schemas import Taxonomy, TaxonomyCategory
from forge.taskfactory.taxonomy import TaxonomyBuilder, TaxonomyError, taxonomy_problems

PROFILE = EnvironmentProfile(
    env_name="mail", env_type="premade:gmail", family="state",
    tools=(ToolInfo(name="/archive_email"), ToolInfo(name="/send_email"), ToolInfo(name="/star_email")),
    state_sample={"emails": []},
)
RUBRIC = {"1": "one action", "3": "several dependent actions", "5": "30+ steps with replanning"}


def _category(name: str, levels: list[int], description: str | None = None, exercises=None) -> TaxonomyCategory:
    return TaxonomyCategory(
        name=name,
        description=description or f"{name} {name}-specific work",
        exercises=exercises if exercises is not None else ["/archive_email"],
        difficulties=levels,
    )


GOOD = Taxonomy(
    categories=[
        _category("triage", [1, 2], "Sort new mail by urgency and flag what needs a reply"),
        _category("follow-up", [2, 3], "Write replies that chase open requests", exercises=["/send_email"]),
        _category(
            "reconciliation", [3, 4, 5], "Match conflicting threads and keep the newest facts",
            exercises=["/star_email", "/archive_email"],
        ),
        _category("cleanup", [1, 4, 5], "Archive stale newsletters across many folders"),
    ],
    difficulty_rubric=RUBRIC,
)


def test_a_diverse_taxonomy_has_no_problems():
    assert taxonomy_problems(GOOD, PROFILE) == []


def test_too_few_categories_is_rejected():
    taxonomy = GOOD.model_copy(update={"categories": GOOD.categories[:3]})

    assert any("4 to 12" in p for p in taxonomy_problems(taxonomy, PROFILE))


def test_covering_fewer_than_three_levels_is_rejected():
    taxonomy = GOOD.model_copy(update={"categories": [_category(n, [1, 2]) for n in ("a", "b", "c", "d")]})

    assert any("3 difficulty" in p for p in taxonomy_problems(taxonomy, PROFILE))


def test_duplicate_category_names_are_rejected():
    categories = [*GOOD.categories[:3], _category("Triage", [5])]
    taxonomy = GOOD.model_copy(update={"categories": categories})

    assert any("triage" in p.lower() for p in taxonomy_problems(taxonomy, PROFILE))


def test_near_identical_descriptions_are_rejected():
    same = "Archive old newsletters from the inbox"
    categories = [*GOOD.categories[:2], _category("x", [4], same), _category("y", [5], same + ".")]
    taxonomy = GOOD.model_copy(update={"categories": categories})

    assert any("overlap" in p for p in taxonomy_problems(taxonomy, PROFILE))


def test_exercising_a_tool_the_environment_lacks_is_rejected():
    categories = [*GOOD.categories[:3], _category("travel", [5], exercises=["/book_flight"])]
    taxonomy = GOOD.model_copy(update={"categories": categories})

    assert any("/book_flight" in p for p in taxonomy_problems(taxonomy, PROFILE))


def test_a_level_outside_one_to_five_is_rejected():
    categories = [*GOOD.categories[:3], _category("z", [6])]
    taxonomy = GOOD.model_copy(update={"categories": categories})

    assert any("1 to 5" in p for p in taxonomy_problems(taxonomy, PROFILE))


class _ScriptedClient:
    def __init__(self, *responses: Taxonomy) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def extract(self, system: str, user: str, schema):
        self.prompts.append(user)
        return self._responses.pop(0)


def test_the_builder_retries_once_with_the_problems_listed():
    bad = GOOD.model_copy(update={"categories": GOOD.categories[:3]})
    client = _ScriptedClient(bad, GOOD)

    assert TaxonomyBuilder(client).build(PROFILE, count=20) == GOOD
    assert "4 to 12" in client.prompts[1]


def test_the_builder_fails_after_the_retry_also_fails():
    bad = GOOD.model_copy(update={"categories": GOOD.categories[:3]})
    client = _ScriptedClient(bad, bad)

    with pytest.raises(TaxonomyError, match="4 to 12"):
        TaxonomyBuilder(client).build(PROFILE, count=20)


def test_the_prompt_names_the_real_tools_and_the_count():
    client = _ScriptedClient(GOOD)

    TaxonomyBuilder(client).build(PROFILE, count=37)

    assert "/archive_email" in client.prompts[0]
    assert "37" in client.prompts[0]
