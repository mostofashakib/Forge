"""The shapes the task factory reads and writes.

One task shape covers every environment type. What differs per type is which
seed fields and check kinds it uses, which `static_check` enforces:

  state    (in-process, custom and premade apps)  seed rows, state checks
  cli                                              setup commands, shell checks
  browser                                          static pages, DOM and URL checks

Checks are one flat model with a `kind`, so the LLM fills a single simple
schema and the per-kind required fields are checked in code.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EnvFamily = Literal["state", "cli", "browser"]

# Golden-solution length per difficulty, inclusive. None means no ceiling.
DIFFICULTY_STEP_BOUNDS: dict[int, tuple[int, int | None]] = {
    1: (1, 5),
    2: (5, 12),
    3: (12, 30),
    4: (30, None),
    5: (30, None),
}
LONG_HORIZON_DIFFICULTY = 4
MIN_REFLECTION_POINTS = 3
MAX_GOLDEN_STEPS = 200

CheckKind = Literal[
    "record_exists",
    "record_absent",
    "record_count",
    "value",
    "actions_called",
    "actions_not_called",
    "shell",
    "dom",
    "url",
]
# Kinds that look at the world after the episode. A task needs at least one,
# and at least one must fail before the golden solution runs.
OUTCOME_KINDS: frozenset[str] = frozenset(
    {"record_exists", "record_absent", "record_count", "value", "shell", "dom", "url"}
)
FAMILY_CHECK_KINDS: dict[str, frozenset[str]] = {
    "state": frozenset(
        {"record_exists", "record_absent", "record_count", "value", "actions_called", "actions_not_called"}
    ),
    "cli": frozenset({"shell"}),
    "browser": frozenset({"dom", "url"}),
}


class TaxonomyCategory(BaseModel):
    name: str
    description: str
    # Tools or actions this category leans on. Empty for CLI and browser,
    # whose tools are not a fixed list.
    exercises: list[str] = Field(default_factory=list)
    difficulties: list[int]


class Taxonomy(BaseModel):
    categories: list[TaxonomyCategory]
    # What difficulty "1", "3" and "5" mean in this environment.
    difficulty_rubric: dict[str, str] = Field(default_factory=dict)


class GoldenStep(BaseModel):
    """One step of the reference solution, in the environment's action format.

    state: tool is the action name (in-process) or endpoint path (apps).
    cli: tool is "shell" and args has "command".
    browser: tool is goto, click, fill, select, press or check.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Check(BaseModel):
    kind: CheckKind
    description: str = ""
    # record_* and value
    collection: str | None = None
    match: dict[str, Any] | None = None
    expect: dict[str, Any] | None = None
    path: str | None = None
    op: str | None = None
    value: Any = None
    # actions_*
    tools: list[str] | None = None
    ordered: bool = False
    # shell
    command: str | None = None
    # dom
    selector: str | None = None
    prop: str | None = None
    attr: str | None = None


class TaskPage(BaseModel):
    path: str
    html: str


class TaskSeed(BaseModel):
    # state: rows merged into the reset state, by collection.
    records: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # cli: shell commands run before the episode.
    setup: list[str] = Field(default_factory=list)
    # browser: pages served at http://task.local, and where the episode opens.
    pages: list[TaskPage] = Field(default_factory=list)
    start_path: str | None = None


class ReflectionPoint(BaseModel):
    """A place in a long task where the agent has to stop and rethink."""

    step: int  # 0-based index into the golden solution
    kind: Literal["discovery", "reconcile", "verify", "backtrack"]
    note: str


class TaskDraft(BaseModel):
    """What the writer returns for one slot."""

    slot: int
    title: str
    objective: str
    seed: TaskSeed = Field(default_factory=TaskSeed)
    golden: list[GoldenStep]
    checks: list[Check]
    reflection_points: list[ReflectionPoint] = Field(default_factory=list)


class TaskDraftBatch(BaseModel):
    tasks: list[TaskDraft]


class ReviewVerdict(BaseModel):
    """The validator's reading of one task."""

    slot: int
    realistic: bool
    realistic_reason: str
    fair: bool
    fair_reason: str
    sensible: bool
    sensible_reason: str

    @property
    def accepted(self) -> bool:
        return self.realistic and self.fair and self.sensible

    def rejection_reason(self) -> str:
        reasons = [
            f"{name}: {reason}"
            for name, ok, reason in (
                ("not realistic", self.realistic, self.realistic_reason),
                ("not fair", self.fair, self.fair_reason),
                ("does not make sense", self.sensible, self.sensible_reason),
            )
            if not ok
        ]
        return "; ".join(reasons)


class ReviewVerdictBatch(BaseModel):
    verdicts: list[ReviewVerdict]


class SyntheticTask(BaseModel):
    """An accepted task, as the registry stores it."""

    id: str
    category: str
    difficulty: int
    title: str
    objective: str
    seed: TaskSeed
    golden: list[GoldenStep]
    checks: list[Check]
    reflection_points: list[ReflectionPoint]
    # Twice the golden length: room to explore without an open-ended budget.
    step_budget: int
    round: int
    review: ReviewVerdict
    # Final-state fingerprint every pass^k run reproduced.
    fingerprint: str
    contamination_report: Any | None = None


class TaskRejection(BaseModel):
    slot: int
    category: str
    difficulty: int
    round: int
    stage: Literal["writer", "static", "pass_k", "review", "contamination"]
    reason: str
    # None when the writer returned nothing for the slot.
    draft: TaskDraft | None = None
