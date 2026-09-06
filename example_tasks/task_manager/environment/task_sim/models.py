"""Task-tracker domain rules and the scenario record types.

Entities (stored as SQLite tables by `service`):

- User: shared actor model indexed by user_id.
- Project: container for tasks indexed by project_id.
- Milestone: timed checkpoint within a project indexed by milestone_id.
- Task: work item indexed by task_id.
- TaskDependency: directed edge (task_id depends_on depends_on_task_id).
- TaskAssignment: task/user relationship indexed by assignment_id.
- TaskAuditEvent: append-only deterministic audit record indexed by event_id.
- Report: what the acting user stated as the outcome of its work.

State machine:

- PENDING     -> IN_PROGRESS | BLOCKED | CANCELLED | DELETED | ARCHIVED | DUPLICATE
- IN_PROGRESS -> BLOCKED | COMPLETED | CANCELLED | DELETED | ARCHIVED
- BLOCKED     -> IN_PROGRESS | CANCELLED | DELETED | ARCHIVED
- COMPLETED   -> DELETED | ARCHIVED
- CANCELLED   -> DELETED | ARCHIVED
- ARCHIVED    is terminal.
- DUPLICATE   is terminal.
- DELETED     is terminal.

Archival semantics:

- DELETED   (deleted=True)  hard-closed; hidden by default; irreversible.
- DUPLICATE (deleted=True)  marked duplicate; hidden by default; irreversible.
- ARCHIVED  (deleted=False) soft-closed; hidden from list_tasks by default;
                            preserved for history; include_archived=True lists it.
- CANCELLED (deleted=False) explicitly stopped; still visible in default lists.

Permission rules:

- Admins can mutate any task.
- Creators can update title, description, assignee, and status.
- Assignees can update status.
- Assignments require creator or admin permissions.
"""

from __future__ import annotations

from dataclasses import dataclass

TaskStatus = str

VALID_STATUS_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    "PENDING":     {"IN_PROGRESS", "BLOCKED", "CANCELLED", "DELETED", "ARCHIVED", "DUPLICATE"},
    "IN_PROGRESS": {"BLOCKED", "COMPLETED", "CANCELLED", "DELETED", "ARCHIVED"},
    "BLOCKED":     {"IN_PROGRESS", "CANCELLED", "DELETED", "ARCHIVED"},
    "COMPLETED":   {"DELETED", "ARCHIVED"},
    "CANCELLED":   {"DELETED", "ARCHIVED"},
    "ARCHIVED":    set(),
    "DUPLICATE":   set(),
    "DELETED":     set(),
}

STATUS_ALIASES: dict[str, TaskStatus] = {
    "todo":        "PENDING",
    "pending":     "PENDING",
    "in_progress": "IN_PROGRESS",
    "in-progress": "IN_PROGRESS",
    "blocked":     "BLOCKED",
    "done":        "COMPLETED",
    "complete":    "COMPLETED",
    "completed":   "COMPLETED",
    "cancelled":   "CANCELLED",
    "canceled":    "CANCELLED",
    "archived":    "ARCHIVED",
    "duplicate":   "DUPLICATE",
    "dup":         "DUPLICATE",
    "deleted":     "DELETED",
}

PRIORITY_VALUES = {"LOW", "MEDIUM", "HIGH", "URGENT"}

#: Statuses that take a task out of active circulation. Reaching one is
#: destructive in a way no reassignment task ever asks for, so the verifier
#: treats an unrequested transition into this set as leaving the task rather
#: than doing it badly.
CLOSING_STATUSES = frozenset({"DELETED", "DUPLICATE", "ARCHIVED", "CANCELLED"})


def normalize_status(status: str) -> TaskStatus:
    normalized = status.strip()
    return STATUS_ALIASES.get(normalized.lower(), normalized.upper())


# ---------------------------------------------------------------------------
# Scenario records
# ---------------------------------------------------------------------------
#
# The tracker world is a task tracker and nothing more. Everything task-shaped
# is data of these three kinds, evaluated generically by `tracker`, so a second
# Harbor task means writing a second `Scenario` rather than editing the world.


@dataclass(frozen=True, slots=True)
class ScenarioEvent:
    """Something that can happen in this world, once.

    `scheduled_step` pins the activation to the seed calendar instead of one
    step after whatever the agent last did, for events the world does on its
    own clock rather than in response to an action.
    """

    event_id: str
    description: str = ""
    scheduled_step: int | None = None


@dataclass(frozen=True, slots=True)
class ScenarioRule:
    """What the agent has to do for one event to happen.

    Five trigger kinds, each gated by `requires_activated` / `requires_pending`,
    which is how a scenario expresses ordering:

    ``observed``      the serialized tool result carried every id in `observed_ids`
    ``field_equals``  a row reached a value: `table`/`row_id`/`field`/`value`
    ``label_present`` a task carries every label in `labels`
    ``tool_called``   one of `tools` completed successfully
    ``all_of``        a pure dependency closure over `requires_activated`
    """

    rule_id: str
    event_id: str
    trigger: str
    observed_ids: tuple[str, ...] = ()
    table: str = ""
    row_id: str = ""
    field: str = ""
    value: str = ""
    labels: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    requires_activated: tuple[str, ...] = ()
    requires_pending: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LatentTask:
    """A work item that does not exist until its event fires.

    Stored in a table of its own so it is absent from every model-facing query
    -- list, get, search, project rollup, counts -- until the world releases it.
    """

    task_id: str
    event_id: str
    title: str
    description: str
    creator_id: str
    assignee_id: str | None
    status: str
    project_id: str | None
    milestone_id: str | None
    priority: str
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LatentDependency:
    """An edge that appears with its event."""

    dep_id: str
    event_id: str
    task_id: str
    depends_on_task_id: str
