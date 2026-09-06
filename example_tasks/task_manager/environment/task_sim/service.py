#!/usr/bin/env python3
"""The task-tracker world: schema, seed, tools, and the record of what happened.

This module owns the database. It runs only in the environment's own container;
the agent's image does not contain it, which is why the agent has no path to the
seed, the scenario ledger, or the answer key derived from them.

`seed_database` drops the workspace and rebuilds it, and it is the only path to
one: `server.serve()` calls it on startup and the RL environment calls it per
episode. An episode never inherits another episode's rows or half-fired events.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from task_sim.clock import VIRTUAL_CLOCK
from task_sim.identity import LOGGED_IN_USER
from task_sim.models import (
    PRIORITY_VALUES,
    VALID_STATUS_TRANSITIONS,
    normalize_status,
)
from task_sim.scenario import Scenario
from task_sim.seed import (
    REASSIGNMENT_SCENARIO,
    TRACKER_ASSIGNMENTS,
    TRACKER_DEPENDENCIES,
    TRACKER_MILESTONES,
    TRACKER_PROJECTS,
    TRACKER_TASKS,
    TRACKER_USERS,
)
from task_sim.sqlite_common import (
    ToolError,
    UnknownToolError,
    connect,
    query_rows as plain_query_rows,
    remove_pycaches,
    storage_errors,
)
from task_sim.tool_definitions import tool_names, validate_tool_payload
from task_sim.tracker import TRIGGER_KINDS

DEFAULT_DB_PATH = Path("/var/lib/tasks/tasks.db")
DEFAULT_SNAPSHOT_PATH = Path("/var/lib/tasks/tasks_seed_snapshot.sql")

TOOL_NAMES: tuple[str, ...] = tool_names()

SCHEMA = """
CREATE TABLE virtual_clock (
    clock_id   INTEGER PRIMARY KEY CHECK (clock_id = 1),
    current_ms INTEGER NOT NULL
);
-- Time is a pure function of the sequence of world mutations, so a write that
-- would stall or rewind it is a defect rather than a state to tolerate.
CREATE TRIGGER virtual_clock_monotonic
BEFORE UPDATE ON virtual_clock
WHEN NEW.current_ms <= OLD.current_ms
BEGIN
    SELECT RAISE(ABORT, 'virtual clock must advance');
END;

CREATE TABLE users (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    email        TEXT NOT NULL,
    role         TEXT NOT NULL,
    team         TEXT NOT NULL,
    handle       TEXT NOT NULL
);
CREATE TABLE projects (
    project_id    TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    owner_id      TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    archived      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE milestones (
    milestone_id  TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(project_id),
    title         TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    due_at_ms     INTEGER,
    created_at_ms INTEGER NOT NULL
);
CREATE TABLE tasks (
    task_id       TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL,
    creator_id    TEXT NOT NULL,
    assignee_id   TEXT,
    status        TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    project_id    TEXT REFERENCES projects(project_id),
    milestone_id  TEXT REFERENCES milestones(milestone_id),
    due_at_ms     INTEGER,
    priority      TEXT NOT NULL DEFAULT 'MEDIUM',
    labels        TEXT NOT NULL DEFAULT '[]',
    deleted       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE assignments (
    assignment_id  TEXT PRIMARY KEY,
    task_id        TEXT NOT NULL,
    user_id        TEXT NOT NULL,
    assigned_by    TEXT NOT NULL,
    assigned_at_ms INTEGER NOT NULL
);
CREATE TABLE task_dependencies (
    dep_id             TEXT PRIMARY KEY,
    task_id            TEXT NOT NULL,
    depends_on_task_id TEXT NOT NULL
);
CREATE TABLE audit_events (
    event_id    TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL,
    actor_id    TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    before_json TEXT NOT NULL,
    after_json  TEXT NOT NULL,
    at_ms       INTEGER NOT NULL
);

-- What the acting user stated as the outcome of its work. The graded answer
-- has to live in the world, because grading never reads the agent's own logs.
CREATE TABLE reports (
    report_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ms            INTEGER NOT NULL,
    actor_id      TEXT NOT NULL,
    task_ids_json TEXT NOT NULL,
    summary       TEXT NOT NULL DEFAULT ''
);

-- The world's own record of what the agent did. It lives here, not in
-- /logs/agent, so trajectory and side-effect checks rest on evidence the
-- graded party could not have authored.
CREATE TABLE action_log (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    ms             INTEGER NOT NULL,
    actor_id       TEXT NOT NULL,
    tool           TEXT NOT NULL,
    task_id        TEXT,
    project_id     TEXT,
    observed_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE integrity_violations (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    ms       INTEGER NOT NULL,
    actor_id TEXT NOT NULL,
    kind     TEXT NOT NULL,
    surface  TEXT NOT NULL,
    detail   TEXT NOT NULL
);

CREATE TABLE scenario_events (
    event_id       TEXT PRIMARY KEY,
    description    TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'pending',
    scheduled_step INTEGER,
    activated_ms   INTEGER
);
CREATE TABLE scenario_rules (
    rule_id            TEXT PRIMARY KEY,
    event_id           TEXT NOT NULL REFERENCES scenario_events(event_id),
    trigger            TEXT NOT NULL,
    observed_ids       TEXT NOT NULL DEFAULT '[]',
    table_name         TEXT,
    row_id             TEXT,
    field_name         TEXT,
    field_value        TEXT,
    labels             TEXT NOT NULL DEFAULT '[]',
    tools              TEXT NOT NULL DEFAULT '[]',
    requires_activated TEXT NOT NULL DEFAULT '[]',
    requires_pending   TEXT NOT NULL DEFAULT '[]'
);
-- Latent rows are absent from the live tables until their event fires, so they
-- cannot affect listings, counts, project rollups or dependency queries.
CREATE TABLE latent_tasks (
    task_id      TEXT PRIMARY KEY,
    event_id     TEXT NOT NULL REFERENCES scenario_events(event_id),
    title        TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    creator_id   TEXT NOT NULL,
    assignee_id  TEXT,
    status       TEXT NOT NULL,
    project_id   TEXT,
    milestone_id TEXT,
    priority     TEXT NOT NULL DEFAULT 'MEDIUM',
    labels       TEXT NOT NULL DEFAULT '[]',
    released     INTEGER NOT NULL DEFAULT 0,
    released_ms  INTEGER
);
CREATE TABLE latent_dependencies (
    dep_id             TEXT PRIMARY KEY,
    event_id           TEXT NOT NULL REFERENCES scenario_events(event_id),
    task_id            TEXT NOT NULL,
    depends_on_task_id TEXT NOT NULL,
    released           INTEGER NOT NULL DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def seed_database(
    db_path: Path,
    snapshot_path: Path,
    scenario: Scenario = REASSIGNMENT_SCENARIO,
) -> None:
    """Drop the workspace and rebuild it. The only path to a workspace."""
    for rule in scenario.rules:
        if rule.trigger not in TRIGGER_KINDS:
            # An authoring mistake, reported now rather than as a rule that
            # silently never fires and an episode nobody can explain.
            raise ValueError(
                f"rule {rule.rule_id} uses unknown trigger {rule.trigger!r}; "
                f"known triggers: {sorted(TRIGGER_KINDS)}"
            )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)
        VIRTUAL_CLOCK.initialize(connection, 0)

        connection.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)", TRACKER_USERS)
        connection.executemany(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)",
            [
                (project_id, name, description, owner_id,
                 VIRTUAL_CLOCK.at(step), 1 if archived else 0)
                for project_id, name, description, owner_id, step, archived in TRACKER_PROJECTS
            ],
        )
        connection.executemany(
            "INSERT INTO milestones VALUES (?, ?, ?, ?, ?, ?)",
            [
                (milestone_id, project_id, title, description, due_at_ms,
                 VIRTUAL_CLOCK.at(step))
                for milestone_id, project_id, title, description, due_at_ms, step in TRACKER_MILESTONES
            ],
        )
        connection.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    task_id, title, description, creator_id, assignee_id, status,
                    VIRTUAL_CLOCK.at(step), VIRTUAL_CLOCK.at(step),
                    project_id, milestone_id, due_at_ms, priority,
                    json.dumps(list(labels)),
                    1 if status in ("DELETED", "DUPLICATE") else 0,
                )
                for (
                    task_id, title, description, creator_id, assignee_id, status,
                    project_id, milestone_id, due_at_ms, priority, labels, step,
                ) in TRACKER_TASKS
            ],
        )
        connection.executemany(
            "INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
            [
                (assignment_id, task_id, user_id, assigned_by, VIRTUAL_CLOCK.at(step))
                for assignment_id, task_id, user_id, assigned_by, step in TRACKER_ASSIGNMENTS
            ],
        )
        connection.executemany(
            "INSERT INTO task_dependencies VALUES (?, ?, ?)", TRACKER_DEPENDENCIES
        )

        connection.executemany(
            "INSERT INTO scenario_events (event_id, description, status, scheduled_step) "
            "VALUES (?, ?, 'pending', ?)",
            [(event.event_id, event.description, event.scheduled_step) for event in scenario.events],
        )
        connection.executemany(
            "INSERT INTO scenario_rules ("
            "  rule_id, event_id, trigger, observed_ids, table_name, row_id,"
            "  field_name, field_value, labels, tools, requires_activated, requires_pending"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    rule.rule_id, rule.event_id, rule.trigger,
                    json.dumps(list(rule.observed_ids)),
                    rule.table or None, rule.row_id or None,
                    rule.field or None, rule.value or None,
                    json.dumps(list(rule.labels)),
                    json.dumps(list(rule.tools)),
                    json.dumps(list(rule.requires_activated)),
                    json.dumps(list(rule.requires_pending)),
                )
                for rule in scenario.rules
            ],
        )
        connection.executemany(
            "INSERT INTO latent_tasks ("
            "  task_id, event_id, title, description, creator_id, assignee_id,"
            "  status, project_id, milestone_id, priority, labels"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    task.task_id, task.event_id, task.title, task.description,
                    task.creator_id, task.assignee_id, task.status,
                    task.project_id, task.milestone_id, task.priority,
                    json.dumps(list(task.labels)),
                )
                for task in scenario.latent_tasks
            ],
        )
        connection.executemany(
            "INSERT INTO latent_dependencies (dep_id, event_id, task_id, depends_on_task_id) "
            "VALUES (?, ?, ?, ?)",
            [
                (item.dep_id, item.event_id, item.task_id, item.depends_on_task_id)
                for item in scenario.latent_dependencies
            ],
        )

        connection.commit()
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text("\n".join(connection.iterdump()), encoding="utf-8")


def teardown_database(
    db_path: Path,
    snapshot_path: Path,
    scenario: Scenario = REASSIGNMENT_SCENARIO,
) -> None:
    remove_pycaches()
    if not snapshot_path.exists():
        seed_database(db_path, snapshot_path, scenario)
        return
    if db_path.exists():
        db_path.unlink()
    with connect(db_path) as connection:
        connection.executescript(snapshot_path.read_text(encoding="utf-8"))
        connection.commit()


def save_snapshot(db_path: Path, snapshot_path: Path) -> None:
    with connect(db_path) as connection:
        snapshot_path.write_text("\n".join(connection.iterdump()), encoding="utf-8")


def query_rows(
    connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    rows = plain_query_rows(connection, sql, params)
    for row in rows:
        if isinstance(row.get("labels"), str):
            try:
                row["labels"] = json.loads(row["labels"])
            except (json.JSONDecodeError, TypeError):
                row["labels"] = []
    return rows


def export_state(db_path: Path) -> dict[str, Any]:
    """Everything a verifier is allowed to read, and nothing the agent wrote.

    Latent rows are reported as a ledger of what is still withheld -- ids and
    their event, never their content -- so a grader can check visibility
    against the event record without the export becoming the answer key it is
    meant to grade against.
    """
    with connect(db_path) as connection:
        return {
            "clock_ms":     VIRTUAL_CLOCK.now(connection),
            "users":        query_rows(connection, "SELECT * FROM users ORDER BY user_id"),
            "projects":     query_rows(connection, "SELECT * FROM projects ORDER BY project_id"),
            "milestones":   query_rows(connection, "SELECT * FROM milestones ORDER BY milestone_id"),
            "tasks":        query_rows(connection, "SELECT * FROM tasks ORDER BY task_id"),
            "assignments":  query_rows(connection, "SELECT * FROM assignments ORDER BY assignment_id"),
            "dependencies": query_rows(connection, "SELECT * FROM task_dependencies ORDER BY dep_id"),
            "audit_events": query_rows(connection, "SELECT * FROM audit_events ORDER BY event_id"),
            "reports":      query_rows(connection, "SELECT * FROM reports ORDER BY report_id"),
            "scenario_events": query_rows(
                connection,
                "SELECT event_id, status, scheduled_step, activated_ms "
                "FROM scenario_events ORDER BY event_id",
            ),
            "latent_tasks": query_rows(
                connection,
                "SELECT task_id, event_id, released, released_ms FROM latent_tasks ORDER BY task_id",
            ),
            "action_log": plain_query_rows(connection, "SELECT * FROM action_log ORDER BY seq"),
            "integrity_violations": plain_query_rows(
                connection, "SELECT * FROM integrity_violations ORDER BY seq"
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _labels_of(row: sqlite3.Row | dict[str, Any]) -> list[str]:
    raw = row["labels"] if not isinstance(row, dict) else row.get("labels")
    if isinstance(raw, list):
        return list(raw)
    try:
        return json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []


def _task_dict(connection: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        raise ToolError("task_not_found", "not_found", "Task was not found.")
    task = dict(row)
    task["labels"] = _labels_of(row)
    return task


def _require_user(connection: sqlite3.Connection, user_id: str, message: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        raise ToolError("user_not_found", "permission_denied", message)
    return row


def insert_audit(
    connection: sqlite3.Connection,
    task_id: str,
    actor_id: str,
    event_type: str,
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, Any]:
    count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    event_id = f"AUDIT{count + 1:03d}"
    at_ms = VIRTUAL_CLOCK.action_instant(connection)
    connection.execute(
        "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            event_id, task_id, actor_id, event_type,
            json.dumps(before, sort_keys=True),
            json.dumps(after, sort_keys=True),
            at_ms,
        ),
    )
    return {
        "event_id": event_id, "task_id": task_id, "actor_id": actor_id,
        "event_type": event_type, "before": before, "after": after, "at_ms": at_ms,
    }


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def list_tasks(
    db_path: Path,
    include_deleted: bool = False,
    include_archived: bool = False,
    project_id: str | None = None,
    status_filter: str | None = None,
    priority_filter: str | None = None,
    milestone_filter: str | None = None,
    assignee_filter: str | None = None,
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if not include_deleted:
        clauses.append("deleted = 0")
    if not include_archived:
        clauses.append("status != 'ARCHIVED'")
    if project_id:
        clauses.append("project_id = ?")
        params.append(project_id)
    if milestone_filter:
        clauses.append("milestone_id = ?")
        params.append(milestone_filter)
    if status_filter:
        clauses.append("status = ?")
        params.append(normalize_status(status_filter))
    if priority_filter:
        clauses.append("priority = ?")
        params.append(priority_filter.upper())
    if assignee_filter:
        clauses.append("assignee_id = ?")
        params.append(assignee_filter)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(db_path) as connection:
        tasks = query_rows(connection, f"SELECT * FROM tasks {where} ORDER BY task_id", tuple(params))
    return {"tasks": tasks, "count": len(tasks)}


def get_task(db_path: Path, task_id: str) -> dict[str, Any]:
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM tasks WHERE task_id = ? AND deleted = 0", (task_id,)
        ).fetchone()
        if row is None:
            raise ToolError("task_not_found", "not_found", "Task was not found.")
        task = dict(row)
        task["labels"] = _labels_of(row)
        depends_on = [
            item["depends_on_task_id"]
            for item in query_rows(
                connection,
                "SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ? ORDER BY dep_id",
                (task_id,),
            )
        ]
        required_by = [
            item["task_id"]
            for item in query_rows(
                connection,
                "SELECT task_id FROM task_dependencies WHERE depends_on_task_id = ? ORDER BY dep_id",
                (task_id,),
            )
        ]
    return {"id": task_id, "task": task, "depends_on": depends_on, "required_by": required_by}


def list_users(db_path: Path) -> dict[str, Any]:
    with connect(db_path) as connection:
        return {"users": query_rows(connection, "SELECT * FROM users ORDER BY user_id")}


def list_projects(db_path: Path, include_archived: bool = False) -> dict[str, Any]:
    where = "" if include_archived else "WHERE archived = 0"
    with connect(db_path) as connection:
        return {"projects": query_rows(connection, f"SELECT * FROM projects {where} ORDER BY project_id")}


def get_project(db_path: Path, project_id: str) -> dict[str, Any]:
    with connect(db_path) as connection:
        project = connection.execute(
            "SELECT * FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone()
        if project is None:
            raise ToolError("project_not_found", "not_found", "Project was not found.")
        tasks = query_rows(
            connection,
            "SELECT * FROM tasks WHERE project_id = ? AND deleted = 0 "
            "AND status != 'ARCHIVED' ORDER BY task_id",
            (project_id,),
        )
        milestones = query_rows(
            connection,
            "SELECT * FROM milestones WHERE project_id = ? ORDER BY milestone_id",
            (project_id,),
        )
    return {"id": project_id, "project": dict(project), "tasks": tasks, "milestones": milestones}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def create_task(
    db_path: Path,
    title: str,
    description: str = "",
    actor_id: str = LOGGED_IN_USER.user_id,
    task_id: str = "",
    assignee_id: str = "",
    status: str = "PENDING",
    project_id: str | None = None,
    milestone_id: str | None = None,
    due_at_ms: int | None = None,
    priority: str = "MEDIUM",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    title = title.strip()
    if not title:
        raise ToolError("invalid_arguments", "validation_error", "Task title is required.")
    status = normalize_status(status)
    if status not in VALID_STATUS_TRANSITIONS:
        raise ToolError("invalid_arguments", "validation_error", "Unknown task status.")
    priority = priority.upper()
    if priority not in PRIORITY_VALUES:
        raise ToolError("invalid_arguments", "validation_error", "Unknown priority value.")
    assignee_id = (assignee_id or "").strip()

    with connect(db_path) as connection:
        _require_user(connection, actor_id, "Acting user was not found.")
        if assignee_id and connection.execute(
            "SELECT 1 FROM users WHERE user_id = ?", (assignee_id,)
        ).fetchone() is None:
            raise ToolError("user_not_found", "not_found", "Assignee user was not found.")
        if connection.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone():
            raise ToolError("duplicate_task", "conflict", "Task id already exists.")
        if not task_id:
            count = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            task_id = f"TASK{count + 1:03d}"
            if connection.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone():
                raise ToolError("duplicate_task", "conflict", "Task id already exists.")

        now_ms = VIRTUAL_CLOCK.action_instant(connection)
        connection.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id, title, description.strip(), actor_id, assignee_id or None,
                status, now_ms, now_ms, project_id, milestone_id, due_at_ms, priority,
                json.dumps(list(labels or [])),
                1 if status in ("DELETED", "DUPLICATE") else 0,
            ),
        )
        if assignee_id:
            count = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
            connection.execute(
                "INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
                (f"ASSIGN{count + 1:03d}", task_id, assignee_id, actor_id, now_ms),
            )
        audit = insert_audit(
            connection, task_id, actor_id, "task_created", {},
            {"status": status, "title": title, "assignee_id": assignee_id},
        )
        connection.commit()
        task = _task_dict(connection, task_id)
    return {"id": task_id, "task": task, "audit_event": audit}


def update_task(
    db_path: Path,
    task_id: str,
    actor_id: str = LOGGED_IN_USER.user_id,
    title: str | None = None,
    description: str | None = None,
    assignee_id: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    project_id: str | None = None,
    milestone_id: str | None = None,
    due_at_ms: int | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    with connect(db_path) as connection:
        task = connection.execute(
            "SELECT * FROM tasks WHERE task_id = ? AND deleted = 0", (task_id,)
        ).fetchone()
        if task is None:
            raise ToolError("task_not_found", "not_found", "Task was not found.")
        actor = _require_user(connection, actor_id, "Acting user was not found.")

        updates: dict[str, Any] = {}
        if title is not None:
            title = title.strip()
            if not title:
                raise ToolError("invalid_arguments", "validation_error", "Task title is required.")
            updates["title"] = title
        if description is not None:
            updates["description"] = description.strip()
        if assignee_id is not None:
            assignee_id = assignee_id.strip()
            if assignee_id and connection.execute(
                "SELECT 1 FROM users WHERE user_id = ?", (assignee_id,)
            ).fetchone() is None:
                raise ToolError("user_not_found", "not_found", "Assignee user was not found.")
            updates["assignee_id"] = assignee_id or None
        if status is not None:
            updates["status"] = normalize_status(status)
        if priority is not None:
            value = priority.upper()
            if value not in PRIORITY_VALUES:
                raise ToolError("invalid_arguments", "validation_error", "Unknown priority value.")
            updates["priority"] = value
        if project_id is not None:
            if connection.execute(
                "SELECT 1 FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone() is None:
                raise ToolError("project_not_found", "not_found", "Project was not found.")
            updates["project_id"] = project_id
        if milestone_id is not None:
            if milestone_id and connection.execute(
                "SELECT 1 FROM milestones WHERE milestone_id = ?", (milestone_id,)
            ).fetchone() is None:
                raise ToolError("milestone_not_found", "not_found", "Milestone was not found.")
            updates["milestone_id"] = milestone_id or None
        if due_at_ms is not None:
            updates["due_at_ms"] = due_at_ms
        if labels is not None:
            updates["labels"] = json.dumps(list(labels))

        metadata_fields = {
            "title", "description", "assignee_id", "priority",
            "project_id", "milestone_id", "labels", "due_at_ms",
        }
        metadata_changed = any(
            field in updates and task[field] != updates[field] for field in metadata_fields
        )
        status_changed = "status" in updates and task["status"] != updates["status"]

        if metadata_changed and actor["role"] != "admin" and task["creator_id"] != actor_id:
            raise ToolError(
                "permission_denied", "permission_denied",
                "Only creators or admins can update task metadata.",
            )
        is_assignee = connection.execute(
            "SELECT 1 FROM assignments WHERE task_id = ? AND user_id = ?", (task_id, actor_id)
        ).fetchone()
        if status_changed and actor["role"] != "admin" and task["creator_id"] != actor_id and not is_assignee:
            raise ToolError(
                "permission_denied", "permission_denied",
                "User cannot update this task status.",
            )
        if "status" in updates:
            new_status = updates["status"]
            if new_status not in VALID_STATUS_TRANSITIONS:
                raise ToolError("invalid_arguments", "validation_error", "Unknown task status.")
            if status_changed and new_status not in VALID_STATUS_TRANSITIONS[task["status"]]:
                raise ToolError(
                    "invalid_status_transition", "state_machine_error",
                    f"Cannot transition task from {task['status']} to {new_status}.",
                )
        if not metadata_changed and not status_changed:
            row = dict(task)
            row["labels"] = _labels_of(task)
            return {"id": task_id, "task": row, "noop": True}

        before: dict[str, str] = {}
        after: dict[str, str] = {}
        for field, value in updates.items():
            if task[field] != value:
                before[field] = "" if task[field] is None else str(task[field])
                after[field] = "" if value is None else str(value)

        now_ms = VIRTUAL_CLOCK.action_instant(connection)
        set_parts = [f"{field} = ?" for field in updates]
        params: list[Any] = list(updates.values())
        set_parts.append("updated_at_ms = ?")
        params.append(now_ms)
        if updates.get("status") in ("DELETED", "DUPLICATE"):
            set_parts.append("deleted = ?")
            params.append(1)
        params.append(task_id)
        connection.execute(
            f"UPDATE tasks SET {', '.join(set_parts)} WHERE task_id = ?", tuple(params)
        )

        if updates.get("assignee_id") and not connection.execute(
            "SELECT 1 FROM assignments WHERE task_id = ? AND user_id = ?",
            (task_id, updates["assignee_id"]),
        ).fetchone():
            count = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
            connection.execute(
                "INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
                (f"ASSIGN{count + 1:03d}", task_id, updates["assignee_id"], actor_id, now_ms),
            )

        event_type = "status_changed" if set(updates) == {"status"} else "task_updated"
        audit = insert_audit(connection, task_id, actor_id, event_type, before, after)
        connection.commit()
        updated = _task_dict(connection, task_id)
    return {"id": task_id, "task": updated, "audit_event": audit}


def delete_task(db_path: Path, task_id: str, actor_id: str = LOGGED_IN_USER.user_id) -> dict[str, Any]:
    return update_task(db_path, task_id=task_id, actor_id=actor_id, status="DELETED")


def archive_task(db_path: Path, task_id: str, actor_id: str = LOGGED_IN_USER.user_id) -> dict[str, Any]:
    return update_task(db_path, task_id=task_id, actor_id=actor_id, status="ARCHIVED")


def mark_task_duplicate(
    db_path: Path, task_id: str, original_task_id: str,
    actor_id: str = LOGGED_IN_USER.user_id,
) -> dict[str, Any]:
    with connect(db_path) as connection:
        task = connection.execute(
            "SELECT * FROM tasks WHERE task_id = ? AND deleted = 0", (task_id,)
        ).fetchone()
        if task is None:
            raise ToolError("task_not_found", "not_found", "Task was not found.")
        actor = _require_user(connection, actor_id, "Acting user was not found.")
        if connection.execute(
            "SELECT 1 FROM tasks WHERE task_id = ?", (original_task_id,)
        ).fetchone() is None:
            raise ToolError("task_not_found", "not_found", "Original task was not found.")
        if task_id == original_task_id:
            raise ToolError(
                "invalid_arguments", "validation_error",
                "A task cannot be a duplicate of itself.",
            )
        if actor["role"] != "admin" and task["creator_id"] != actor_id:
            raise ToolError(
                "permission_denied", "permission_denied",
                "Only creators or admins can mark tasks as duplicates.",
            )
        now_ms = VIRTUAL_CLOCK.action_instant(connection)
        connection.execute(
            "UPDATE tasks SET status = 'DUPLICATE', deleted = 1, updated_at_ms = ? WHERE task_id = ?",
            (now_ms, task_id),
        )
        audit = insert_audit(
            connection, task_id, actor_id, "task_marked_duplicate",
            {"status": task["status"]},
            {"status": "DUPLICATE", "original_task_id": original_task_id},
        )
        connection.commit()
        updated = _task_dict(connection, task_id)
    return {"id": task_id, "task": updated, "audit_event": audit, "original_task_id": original_task_id}


def move_task_to_project(
    db_path: Path, task_id: str, project_id: str,
    milestone_id: str | None = None, actor_id: str = LOGGED_IN_USER.user_id,
) -> dict[str, Any]:
    with connect(db_path) as connection:
        task = connection.execute(
            "SELECT * FROM tasks WHERE task_id = ? AND deleted = 0", (task_id,)
        ).fetchone()
        if task is None:
            raise ToolError("task_not_found", "not_found", "Task was not found.")
        actor = _require_user(connection, actor_id, "Acting user was not found.")
        if connection.execute(
            "SELECT 1 FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone() is None:
            raise ToolError("project_not_found", "not_found", "Project was not found.")
        if actor["role"] != "admin" and task["creator_id"] != actor_id:
            raise ToolError(
                "permission_denied", "permission_denied",
                "Only creators or admins can move tasks.",
            )
        if milestone_id:
            milestone = connection.execute(
                "SELECT * FROM milestones WHERE milestone_id = ?", (milestone_id,)
            ).fetchone()
            if milestone is None:
                raise ToolError("milestone_not_found", "not_found", "Milestone was not found.")
            if milestone["project_id"] != project_id:
                raise ToolError(
                    "invalid_arguments", "validation_error",
                    "Milestone does not belong to the target project.",
                )
        now_ms = VIRTUAL_CLOCK.action_instant(connection)
        connection.execute(
            "UPDATE tasks SET project_id = ?, milestone_id = ?, updated_at_ms = ? WHERE task_id = ?",
            (project_id, milestone_id, now_ms, task_id),
        )
        audit = insert_audit(
            connection, task_id, actor_id, "task_moved",
            {"project_id": task["project_id"] or "", "milestone_id": task["milestone_id"] or ""},
            {"project_id": project_id, "milestone_id": milestone_id or ""},
        )
        connection.commit()
        updated = _task_dict(connection, task_id)
    return {"id": task_id, "task": updated, "audit_event": audit}


def create_project(
    db_path: Path, name: str, description: str = "",
    actor_id: str = LOGGED_IN_USER.user_id, project_id: str = "",
) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise ToolError("invalid_arguments", "validation_error", "Project name is required.")
    with connect(db_path) as connection:
        _require_user(connection, actor_id, "Acting user was not found.")
        if project_id and connection.execute(
            "SELECT 1 FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone():
            raise ToolError("duplicate_project", "conflict", "Project id already exists.")
        if not project_id:
            count = connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            project_id = f"P{count + 1:03d}"
        now_ms = VIRTUAL_CLOCK.action_instant(connection)
        connection.execute(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, name, description.strip(), actor_id, now_ms, 0),
        )
        connection.commit()
        project = dict(
            connection.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        )
    return {"id": project_id, "project": project}


def create_milestone(
    db_path: Path, project_id: str, title: str, description: str = "",
    due_at_ms: int | None = None, actor_id: str = LOGGED_IN_USER.user_id,
    milestone_id: str = "",
) -> dict[str, Any]:
    title = title.strip()
    if not title:
        raise ToolError("invalid_arguments", "validation_error", "Milestone title is required.")
    with connect(db_path) as connection:
        _require_user(connection, actor_id, "Acting user was not found.")
        if connection.execute(
            "SELECT 1 FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone() is None:
            raise ToolError("project_not_found", "not_found", "Project was not found.")
        if milestone_id and connection.execute(
            "SELECT 1 FROM milestones WHERE milestone_id = ?", (milestone_id,)
        ).fetchone():
            raise ToolError("duplicate_milestone", "conflict", "Milestone id already exists.")
        if not milestone_id:
            count = connection.execute("SELECT COUNT(*) FROM milestones").fetchone()[0]
            milestone_id = f"M{count + 1:03d}"
        now_ms = VIRTUAL_CLOCK.action_instant(connection)
        connection.execute(
            "INSERT INTO milestones VALUES (?, ?, ?, ?, ?, ?)",
            (milestone_id, project_id, title, description.strip(), due_at_ms, now_ms),
        )
        connection.commit()
        milestone = dict(
            connection.execute(
                "SELECT * FROM milestones WHERE milestone_id = ?", (milestone_id,)
            ).fetchone()
        )
    return {"id": milestone_id, "milestone": milestone}


def link_tasks(
    db_path: Path, task_id: str, depends_on_task_id: str,
    actor_id: str = LOGGED_IN_USER.user_id,
) -> dict[str, Any]:
    with connect(db_path) as connection:
        for identifier in (task_id, depends_on_task_id):
            if connection.execute(
                "SELECT 1 FROM tasks WHERE task_id = ? AND deleted = 0", (identifier,)
            ).fetchone() is None:
                raise ToolError("task_not_found", "not_found", f"Task {identifier} was not found.")
        if task_id == depends_on_task_id:
            raise ToolError(
                "invalid_arguments", "validation_error", "A task cannot depend on itself."
            )
        existing = connection.execute(
            "SELECT * FROM task_dependencies WHERE task_id = ? AND depends_on_task_id = ?",
            (task_id, depends_on_task_id),
        ).fetchone()
        if existing:
            return {"id": existing["dep_id"], "dependency": dict(existing), "noop": True}
        count = connection.execute("SELECT COUNT(*) FROM task_dependencies").fetchone()[0]
        dep_id = f"DEP{count + 1:03d}"
        VIRTUAL_CLOCK.action_instant(connection)
        connection.execute(
            "INSERT INTO task_dependencies VALUES (?, ?, ?)",
            (dep_id, task_id, depends_on_task_id),
        )
        connection.commit()
    return {
        "id": dep_id,
        "dependency": {
            "dep_id": dep_id, "task_id": task_id, "depends_on_task_id": depends_on_task_id
        },
    }


def unlink_tasks(
    db_path: Path, task_id: str, depends_on_task_id: str,
    actor_id: str = LOGGED_IN_USER.user_id,
) -> dict[str, Any]:
    with connect(db_path) as connection:
        dependency = connection.execute(
            "SELECT dep_id FROM task_dependencies WHERE task_id = ? AND depends_on_task_id = ?",
            (task_id, depends_on_task_id),
        ).fetchone()
        if dependency is None:
            raise ToolError("dependency_not_found", "not_found", "Dependency was not found.")
        VIRTUAL_CLOCK.action_instant(connection)
        connection.execute(
            "DELETE FROM task_dependencies WHERE dep_id = ?", (dependency["dep_id"],)
        )
        connection.commit()
    return {"removed": True, "task_id": task_id, "depends_on_task_id": depends_on_task_id}


def submit_handover_report(
    db_path: Path, task_ids: list[str], summary: str = "",
    actor_id: str = LOGGED_IN_USER.user_id,
) -> dict[str, Any]:
    """Record what the acting user says it did.

    Appended rather than overwritten: an episode that reports twice made two
    statements, and which one it stood behind is the last one. Keeping both
    means a grader can see a correction as a correction instead of as the only
    thing that was ever said.
    """
    if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
        raise ToolError(
            "invalid_arguments", "validation_error", "task_ids must be a list of strings."
        )
    cleaned = [item.strip() for item in task_ids if item.strip()]
    with connect(db_path) as connection:
        _require_user(connection, actor_id, "Acting user was not found.")
        now_ms = VIRTUAL_CLOCK.action_instant(connection)
        connection.execute(
            "INSERT INTO reports (ms, actor_id, task_ids_json, summary) VALUES (?, ?, ?, ?)",
            (now_ms, actor_id, json.dumps(cleaned), str(summary or "").strip()),
        )
        connection.commit()
        report_id = connection.execute(
            "SELECT MAX(report_id) FROM reports"
        ).fetchone()[0]
    return {"id": report_id, "report": {"task_ids": cleaned, "summary": summary, "at_ms": now_ms}}


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
#
# Leaving the environment is not a move. These record the attempt whether or
# not it worked -- and mostly it cannot, because the grading directory is
# root-only and the privileged socket is 0600. What is worth keeping is that it
# was made, because that is the part a state export otherwise loses entirely.

_ESCAPE_MARKERS = (
    "/opt/grading", "/var/lib/tasks", "/logs/verifier", "/run/tasks/admin",
    "../", "..\\", "/etc/passwd", "/proc/self",
)
#: SQL that only makes sense if the sender is aiming at the store itself.
_SQL_MARKERS = (
    "drop table", "delete from", "attach database", "detach database",
    "insert into", "update ", "union select", "pragma ", "; --",
)
#: Fields that say who acted. The actor is decided by the socket a request
#: arrived on and is never taken from a payload, so naming one is an attempt.
_ACTOR_FIELDS = ("actor_id", "as_user", "author_id", "impersonate", "run_as", "on_behalf_of")
#: The genuinely free-text fields. A task called "Delete from staging" is a task,
#: and charging an episode for the words in a title would make the clearest
#: description the most expensive one to write.
_PROSE_FIELDS = ("title", "description", "summary")


def _scan_payload(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Attempts visible in one tool call's arguments."""
    found: list[tuple[str, str]] = []
    for field in _ACTOR_FIELDS:
        if field in payload:
            found.append(("actor_override", f"{field}={payload[field]!r}"))
            break

    def scan(key: str, value: Any) -> None:
        if key in _PROSE_FIELDS or not isinstance(value, str):
            return
        lowered = value.lower()
        for marker in _ESCAPE_MARKERS:
            if marker in lowered:
                found.append(("sandbox_escape", f"{key}={value[:120]!r}"))
                return
        for marker in _SQL_MARKERS:
            if marker in lowered:
                found.append(("raw_sql", f"{key}={value[:120]!r}"))
                return

    for key, value in payload.items():
        if isinstance(value, list):
            for item in value:
                scan(key, item)
        else:
            scan(key, value)
    return found


def record_integrity_violation(
    db_path: Path, kind: str, detail: str, surface: str = "agent_socket",
    actor_id: str = LOGGED_IN_USER.user_id,
) -> None:
    """Append an attempt to the world's own record."""
    with connect(db_path) as connection:
        connection.execute(
            "INSERT INTO integrity_violations (ms, actor_id, kind, surface, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (VIRTUAL_CLOCK.now(connection), actor_id, kind, surface, detail),
        )
        connection.commit()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _labels_argument(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


_TOOL_HANDLERS: dict[str, Any] = {
    "list_tasks": lambda db, payload, actor: list_tasks(
        db,
        include_deleted=bool(payload.get("include_deleted", False)),
        include_archived=bool(payload.get("include_archived", False)),
        project_id=payload.get("project_id"),
        status_filter=payload.get("status"),
        priority_filter=payload.get("priority"),
        milestone_filter=payload.get("milestone_id"),
        assignee_filter=payload.get("assignee"),
    ),
    "get_task": lambda db, payload, actor: get_task(db, str(payload.get("task_id", ""))),
    "list_users": lambda db, payload, actor: list_users(db),
    "list_projects": lambda db, payload, actor: list_projects(
        db, bool(payload.get("include_archived", False))
    ),
    "get_project": lambda db, payload, actor: get_project(db, str(payload.get("project_id", ""))),
    "create_task": lambda db, payload, actor: create_task(
        db,
        title=str(payload.get("title", "")),
        description=str(payload.get("description", "")),
        actor_id=actor,
        task_id=payload.get("task_id") or "",
        assignee_id=payload.get("assignee") or "",
        status=payload.get("status") or "PENDING",
        project_id=payload.get("project_id"),
        milestone_id=payload.get("milestone_id"),
        due_at_ms=payload.get("due_at_ms"),
        priority=payload.get("priority") or "MEDIUM",
        labels=_labels_argument(payload.get("labels")),
    ),
    "update_task": lambda db, payload, actor: update_task(
        db,
        task_id=str(payload.get("task_id", "")),
        actor_id=actor,
        title=payload.get("title"),
        description=payload.get("description"),
        assignee_id=payload.get("assignee"),
        status=payload.get("status"),
        priority=payload.get("priority"),
        project_id=payload.get("project_id"),
        milestone_id=payload.get("milestone_id"),
        due_at_ms=payload.get("due_at_ms"),
        labels=_labels_argument(payload.get("labels")),
    ),
    "delete_task": lambda db, payload, actor: delete_task(
        db, str(payload.get("task_id", "")), actor
    ),
    "archive_task": lambda db, payload, actor: archive_task(
        db, str(payload.get("task_id", "")), actor
    ),
    "mark_task_duplicate": lambda db, payload, actor: mark_task_duplicate(
        db,
        task_id=str(payload.get("task_id", "")),
        original_task_id=str(payload.get("original_task_id", "")),
        actor_id=actor,
    ),
    "move_task_to_project": lambda db, payload, actor: move_task_to_project(
        db,
        task_id=str(payload.get("task_id", "")),
        project_id=str(payload.get("project_id", "")),
        milestone_id=payload.get("milestone_id"),
        actor_id=actor,
    ),
    "create_project": lambda db, payload, actor: create_project(
        db,
        name=str(payload.get("name", "")),
        description=str(payload.get("description", "")),
        actor_id=actor,
        project_id=payload.get("project_id") or "",
    ),
    "create_milestone": lambda db, payload, actor: create_milestone(
        db,
        project_id=str(payload.get("project_id", "")),
        title=str(payload.get("title", "")),
        description=str(payload.get("description", "")),
        due_at_ms=payload.get("due_at_ms"),
        actor_id=actor,
        milestone_id=payload.get("milestone_id") or "",
    ),
    "link_tasks": lambda db, payload, actor: link_tasks(
        db, str(payload.get("task_id", "")), str(payload.get("depends_on_task_id", "")), actor
    ),
    "unlink_tasks": lambda db, payload, actor: unlink_tasks(
        db, str(payload.get("task_id", "")), str(payload.get("depends_on_task_id", "")), actor
    ),
    "submit_handover_report": lambda db, payload, actor: submit_handover_report(
        db,
        task_ids=payload.get("task_ids") or [],
        summary=str(payload.get("summary", "")),
        actor_id=actor,
    ),
}


def execute_tool(
    db_path: Path,
    tool_name: str,
    input_payload: dict[str, Any],
    actor_id: str = LOGGED_IN_USER.user_id,
    trace: Any = None,
) -> dict[str, Any]:
    """Run one tool and let the world react to it.

    `trace` collects why each scenario rule matched or did not. It is a
    world-side diagnostic: the socket server and both clients never pass one,
    and it is never merged into the result the agent receives.
    """
    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise UnknownToolError(tool_name)

    # One boundary for storage faults, around the whole call rather than each
    # cursor: a tool that half-ran and then lost the database has still failed
    # for a storage reason, and the caller needs to hear that and not a
    # rule-shaped refusal it might act on.
    with storage_errors():
        # Recorded before the call runs, so an attempt is kept even when the
        # call carrying it fails -- which is the usual case, and does not make
        # the attempt less of one.
        for kind, detail in _scan_payload(input_payload):
            record_integrity_violation(db_path, kind, f"{tool_name}: {detail}", actor_id=actor_id)

        validation_error = validate_tool_payload(tool_name, input_payload)
        if validation_error:
            raise ToolError("invalid_arguments", "validation_error", validation_error)

        result = handler(db_path, input_payload, actor_id)

        # Evaluated against the serialized result the agent actually received.
        # A handler may inspect more rows internally; an unreturned row cannot
        # have been observed.
        from task_sim.tracker import track_action

        with connect(db_path) as connection:
            track_action(connection, tool_name, input_payload, result, actor_id, trace=trace)
            connection.commit()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-tracker world (privileged).")
    parser.add_argument("command", choices=["seed", "teardown", "state", "save-snapshot", "execute_tool"])
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--tool-name", default="")
    parser.add_argument("--input-payload", default="{}")
    args = parser.parse_args()

    db_path = Path(args.db)
    snapshot_path = Path(args.snapshot)
    if args.command == "seed":
        seed_database(db_path, snapshot_path)
        print(json.dumps({"seeded": True}))
    elif args.command == "teardown":
        teardown_database(db_path, snapshot_path)
        print(json.dumps({"teardown": True}))
    elif args.command == "save-snapshot":
        save_snapshot(db_path, snapshot_path)
        print(json.dumps({"snapshot": str(snapshot_path)}))
    elif args.command == "state":
        print(json.dumps(export_state(db_path), indent=2, sort_keys=True))
    else:
        if not args.tool_name:
            parser.error("execute_tool requires --tool-name")
        result = execute_tool(db_path, args.tool_name, json.loads(args.input_payload))
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
