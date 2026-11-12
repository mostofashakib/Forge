"""Step 2: write one executable task per slot, five slots per LLM call."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from forge.extraction.llm_client import LLMClient
from forge.taskfactory.profile import EnvironmentProfile
from forge.taskfactory.schemas import (
    DIFFICULTY_STEP_BOUNDS,
    LONG_HORIZON_DIFFICULTY,
    MIN_REFLECTION_POINTS,
    TaskDraft,
    TaskDraftBatch,
    Taxonomy,
)
from forge.taskfactory.slots import Slot

# Five long tasks fit well inside WRITER_MAX_TOKENS.
CHUNK_SIZE = 5
WRITER_MAX_TOKENS = 16_000

_COMMON = """\
You write realistic, executable tasks for a reinforcement-learning environment.
Each task fills one SLOT and must match its category and difficulty. Return one
task per slot, with `slot` set to the slot number.

Every task has:
  title, objective: the objective reads like a real user's request. It names
    every fact the agent needs that it cannot find in the environment.
  seed: the starting data the task needs (format below).
  golden: the reference solution, an ordered list of {tool, args} steps that
    solves the task from the seeded start. It must pass every time it runs.
  checks: machine-checkable conditions on the final result. At least one must
    fail before the golden solution runs and every one must pass after it.
  reflection_points: for difficulty 4 and 5, at least %(min_reflections)d places
    (0-based golden step index) where the agent has to stop and rethink: kind
    "discovery" (something found midway changes the plan), "reconcile" (stale
    or conflicting records), "verify" (a result must be confirmed before going
    on) or "backtrack" (a dead end to back out of). Seed the data that makes
    each one real.

  edge_case, edge_case_seed: only for a slot that names an edge case. Set
    edge_case to that name, plant the condition in the seed, and list in
    edge_case_seed the seed elements that plant it (format below). The golden
    solution must handle it. Leave both empty for every other slot.

The golden solution's length must fit the slot's difficulty exactly.
"""

_FAMILY = {
    "state": """\
seed.records: rows to merge into the reset state, keyed by collection name
  from the seedable state. A row with an existing id updates that row, a new
  id adds one. Use stable ids you choose, so golden steps can refer to them.
golden: tool is a tool name from the list, args is its JSON body. For a path
  like /items/{item_id}/close, fill the id into the path.
check kinds:
  record_exists {collection, match, expect}   a row matching `match` has `expect`
  record_absent {collection, match}
  record_count  {collection, match, op, value}  op is ==, !=, >, >=, <, <=
  value         {path, op, value}   dotted path like inbox_unread or a.0.b, op adds contains
  actions_called {tools, ordered}   actions_not_called {tools}
edge_case_seed: "collection:id" for each seed row that plants the edge case.
  Collections and paths refer to the reset state shown above, the same shape
  the seed rows merge into. Use only field names you see there. A collection
  that is empty at reset (often a log or history) has fields you cannot see,
  so never check it: check the records the actions change, and use
  actions_called for which actions ran.
""",
    "cli": """\
seed.setup: shell commands that build the starting files and state.
golden: every step is {"tool": "shell", "args": {"command": "..."}} and must
  exit 0.
checks: kind "shell" with a `command` that exits 0 when the condition holds.
  Checks run in a BusyBox sh with BusyBox's grep, sed, awk, find, stat, sort
  and test. GNU-only flags fail there: find has no -printf, so use
  `stat -c '%s %n'` or `wc -c` for sizes. Golden steps run in bash with GNU
  coreutils.
edge_case_seed: "setup:N" for each setup command that plants the edge case,
  counting from 1.
""",
    "browser": """\
seed.pages: 1 to 3 static HTML pages, each {path, html}, served at
  http://task.local<path>. Use inline JavaScript for behaviour, nothing
  external. seed.start_path is the page the episode opens on.
golden: browser steps: goto {path}, click {selector}, fill {selector, value},
  select {selector, value}, press {selector, key}, check {selector}.
check kinds:
  dom {selector, prop, attr, op, value}  prop is text, value or attr; op is
    equals, contains, exists or absent
  url {op, value}   op is equals or contains, compared with the page path
    and query, like /thanks?order=7
edge_case_seed: the path of each page that plants the edge case.
""",
}


_EDGE_CASES = {
    "boundary_conditions": (
        "the seed holds a boundary value the task must handle correctly: an empty "
        "file or list, a zero or maximum value, or a value exactly at a limit the "
        "objective names."
    ),
    "missing_deps": (
        "something the main work depends on is missing at the start (a directory, "
        "a config file, a referenced record), so the agent has to notice it and "
        "create it first."
    ),
    "conflicting_state": (
        "the seed leaves the work partly done or in the way: a file or record "
        "already exists, a stale value contradicts the request, or an earlier "
        "attempt stopped halfway. The agent has to detect it and resolve it."
    ),
    "recovery": (
        "the seed sets a trap, so the most obvious approach fails in a visible way "
        "and the agent has to read the failure and take another path. Every golden "
        "step must still succeed, so the golden solution takes the working path."
    ),
}


# A concrete length to write toward. Models undershoot a bare range.
_STEP_TARGETS = {1: 3, 2: 8, 3: 20, 4: 36, 5: 45}


_REVISE = """\
REVISION: each slot below comes with your earlier draft, whose golden solution
has the wrong number of steps. Return a reworked task for each slot.
If the solution is too short, widen the task until it needs about the target
number of steps: more records to handle, more sub-goals that depend on each
other, results to verify before going on. Seed the data the new scope needs
and add checks for it. Never pad with repeated, pointless or no-op steps.
If it is too long, narrow the task to fewer sub-goals instead.
"""


def _span(difficulty: int) -> str:
    low, high = DIFFICULTY_STEP_BOUNDS[difficulty]
    span = f"{low} or more" if high is None else f"{low} to {high}"
    return f"{span} (aim for about {_STEP_TARGETS[difficulty]})"


def _edge_label(slot: Slot) -> str:
    return f" | edge case {slot.edge_case}" if slot.edge_case else ""


def _edge_case_lines(slots: Sequence[Slot]) -> list[str]:
    """Describe each edge case the slots name, once."""
    named = sorted({slot.edge_case for slot in slots if slot.edge_case})
    if not named:
        return []
    return ["", "Edge cases to plant:", *[f"  {name}: {_EDGE_CASES[name]}" for name in named]]


class TaskWriter:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def write(
        self,
        profile: EnvironmentProfile,
        taxonomy: Taxonomy,
        slots: Sequence[Slot],
        *,
        avoid_titles: Sequence[str] = (),
        feedback: Mapping[int, str] | None = None,
    ) -> dict[int, TaskDraft]:
        """Return the drafts the writer produced, keyed by slot index."""
        system = _COMMON % {"min_reflections": MIN_REFLECTION_POINTS} + "\n" + _FAMILY[profile.family]
        titles = list(avoid_titles)
        drafts: dict[int, TaskDraft] = {}
        for start in range(0, len(slots), CHUNK_SIZE):
            chunk = slots[start:start + CHUNK_SIZE]
            user = self._prompt(profile, taxonomy, chunk, titles, feedback or {})
            batch: TaskDraftBatch = self._client.extract(system=system, user=user, schema=TaskDraftBatch)
            wanted = {slot.index for slot in chunk}
            for draft in batch.tasks:
                if draft.slot in wanted and draft.slot not in drafts:
                    drafts[draft.slot] = draft
                    titles.append(draft.title)
        return drafts

    def revise(
        self,
        profile: EnvironmentProfile,
        taxonomy: Taxonomy,
        items: Sequence[tuple[Slot, TaskDraft, str]],
    ) -> dict[int, TaskDraft]:
        """Rework drafts whose golden solution is the wrong length for their slot."""
        system = (
            _COMMON % {"min_reflections": MIN_REFLECTION_POINTS} + "\n" + _FAMILY[profile.family] + "\n" + _REVISE
        )
        revised: dict[int, TaskDraft] = {}
        for start in range(0, len(items), CHUNK_SIZE):
            chunk = items[start:start + CHUNK_SIZE]
            lines = [profile.prompt_view(), "", f"Difficulty rubric: {json.dumps(taxonomy.difficulty_rubric, sort_keys=True)}", ""]
            for slot, draft, problem in chunk:
                lines += [
                    f"SLOT {slot.index} | category {slot.category} | difficulty {slot.difficulty} | "
                    f"golden solution of {_span(slot.difficulty)} steps{_edge_label(slot)}",
                    f"  Problem: {problem}",
                    f"  Draft: {draft.model_dump_json(exclude={'slot'})}",
                ]
            lines += _edge_case_lines([slot for slot, _draft, _problem in chunk])
            batch: TaskDraftBatch = self._client.extract(system=system, user="\n".join(lines), schema=TaskDraftBatch)
            wanted = {slot.index for slot, _draft, _problem in chunk}
            for draft in batch.tasks:
                if draft.slot in wanted and draft.slot not in revised:
                    revised[draft.slot] = draft
        return revised

    @staticmethod
    def _prompt(
        profile: EnvironmentProfile,
        taxonomy: Taxonomy,
        chunk: Sequence[Slot],
        titles: Sequence[str],
        feedback: Mapping[int, str],
    ) -> str:
        categories = {c.name: c for c in taxonomy.categories}
        lines = [
            profile.prompt_view(),
            "",
            f"Difficulty rubric: {json.dumps(taxonomy.difficulty_rubric, sort_keys=True)}",
            "",
        ]
        for slot in chunk:
            category = categories.get(slot.category)
            description = category.description if category else ""
            long_horizon = " long horizon with reflection points" if slot.difficulty >= LONG_HORIZON_DIFFICULTY else ""
            lines.append(
                f"SLOT {slot.index} | category {slot.category}: {description} | difficulty "
                f"{slot.difficulty} | golden solution of {_span(slot.difficulty)} steps{long_horizon}"
                f"{_edge_label(slot)}"
            )
            if slot.index in feedback:
                lines.append(f"  A previous task for this slot was rejected: {feedback[slot.index]}")
        lines += _edge_case_lines(chunk)
        if titles:
            lines += ["", "Titles already in this batch (write different tasks):", *[f"  - {t}" for t in titles]]
        return "\n".join(lines)
