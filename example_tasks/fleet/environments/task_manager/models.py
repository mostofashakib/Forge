"""Task manager domain rules shared with the SQLite service.

Entities (stored as SQLite tables by sqlite_service):
- User: shared actor model indexed by user_id.
- Project: container for tasks indexed by project_id.
- Milestone: timed checkpoint within a project indexed by milestone_id.
- Task: work item indexed by task_id.
- TaskDependency: directed dependency edge (task_id depends_on depends_on_task_id).
- TaskAssignment: task/user relationship indexed by assignment_id.
- TaskAuditEvent: append-only deterministic audit record indexed by event_id.

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
- DELETED   (deleted=True)  — hard-closed; hidden by default; irreversible.
- DUPLICATE (deleted=True)  — marked duplicate; hidden by default; irreversible.
- ARCHIVED  (deleted=False) — soft-closed; hidden by default from list_tasks;
                              preserved for history; use include_archived=True to list.
- CANCELLED (deleted=False) — explicitly stopped; still visible in default list.

Permission rules:
- Admins can mutate any task.
- Creators can update title, description, assignee, and status.
- Assignees can update status.
- Assignments require creator or admin permissions.
"""

from __future__ import annotations


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


def normalize_status(status: str) -> TaskStatus:
    normalized = status.strip()
    return STATUS_ALIASES.get(normalized.lower(), normalized.upper())
