"""Step 1: a diverse map of task categories across difficulty levels."""
from __future__ import annotations

import re

from forge.extraction.llm_client import LLMClient
from forge.taskfactory.profile import EnvironmentProfile
from forge.taskfactory.schemas import DIFFICULTY_STEP_BOUNDS, Taxonomy

MIN_CATEGORIES = 4
MAX_CATEGORIES = 12
MIN_LEVELS = 3
# Two descriptions sharing this share of their words describe one category.
_OVERLAP = 0.8

_SYSTEM = """\
You design the task taxonomy for a reinforcement-learning environment. The
taxonomy decides which kinds of tasks a batch will contain, so it must be
diverse and realistic: the work a real user of this product does.

Return 4 to 12 categories. Each category has a short name, a description of
the kind of work and the reasoning it demands, the tools it exercises (exact
names from the tool list, when one is given), and the difficulty levels (1 to
5) it covers. Together the categories must cover at least 3 levels, and no two
may describe the same work.

Difficulty is set by the length of the reference solution:
  1: 1 to 5 steps    2: 5 to 12 steps    3: 12 to 30 steps
  4 and 5: 30 or more steps, long horizon, where the agent has to stop and
  rethink several times (something found midway changes the plan, stale or
  conflicting records must be reconciled, a result must be verified before
  going on, a dead end must be backed out of).

Also return a difficulty_rubric with keys "1", "3" and "5" saying what those
levels mean in this environment specifically.
"""


class TaxonomyError(RuntimeError):
    pass


class TaxonomyBuilder:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def build(self, profile: EnvironmentProfile, count: int) -> Taxonomy:
        """Ask for a taxonomy, check it, retry once with the problems listed."""
        base = f"{profile.prompt_view()}\n\nThe batch will contain {count} tasks."
        taxonomy = self._client.extract(system=_SYSTEM, user=base, schema=Taxonomy)
        problems = taxonomy_problems(taxonomy, profile)
        if not problems:
            return taxonomy
        retry = base + "\n\nYour previous taxonomy had these problems. Fix all of them:\n" + "\n".join(
            f"- {p}" for p in problems
        )
        taxonomy = self._client.extract(system=_SYSTEM, user=retry, schema=Taxonomy)
        problems = taxonomy_problems(taxonomy, profile)
        if problems:
            raise TaxonomyError("taxonomy still invalid after one retry: " + "; ".join(problems))
        return taxonomy


def taxonomy_problems(taxonomy: Taxonomy, profile: EnvironmentProfile) -> list[str]:
    categories = taxonomy.categories
    problems: list[str] = []
    if not MIN_CATEGORIES <= len(categories) <= MAX_CATEGORIES:
        problems.append(f"needs {MIN_CATEGORIES} to {MAX_CATEGORIES} categories, got {len(categories)}")
    levels = {level for c in categories for level in c.difficulties}
    bad_levels = sorted(level for level in levels if level not in DIFFICULTY_STEP_BOUNDS)
    if bad_levels:
        problems.append(f"difficulty levels must be 1 to 5, got {bad_levels}")
    if len(levels & set(DIFFICULTY_STEP_BOUNDS)) < MIN_LEVELS:
        problems.append(f"categories must cover at least {MIN_LEVELS} difficulty levels, got {sorted(levels)}")
    problems += [f"category {c.name!r} covers no difficulty level" for c in categories if not c.difficulties]
    seen: dict[str, str] = {}
    for category in categories:
        key = category.name.strip().lower()
        if key in seen:
            problems.append(f"category name {category.name!r} is used twice")
        seen[key] = category.name
    problems += _overlap_problems(taxonomy)
    if profile.family == "state" and profile.tools:
        problems += [
            f"category {c.name!r} exercises {tool!r}, which the environment does not have"
            for c in categories
            for tool in c.exercises
            if not profile.has_tool(tool)
        ]
    return problems


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap_problems(taxonomy: Taxonomy) -> list[str]:
    problems = []
    categories = taxonomy.categories
    for i, first in enumerate(categories):
        for second in categories[i + 1:]:
            a, b = _words(first.description), _words(second.description)
            if a and b and len(a & b) / len(a | b) >= _OVERLAP:
                problems.append(f"categories {first.name!r} and {second.name!r} overlap")
    return problems
