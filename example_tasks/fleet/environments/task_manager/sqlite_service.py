#!/usr/bin/env python3
"""SQLite-backed Task Manager service used by Harbor task environments."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from fleet.environments.sqlite_common import ToolError, connect, remove_pycaches
from fleet.environments.sqlite_common import query_rows as plain_query_rows
from fleet.environments.task_manager.models import (
    PRIORITY_VALUES,
    VALID_STATUS_TRANSITIONS,
    normalize_status,
)
from fleet.environments.task_manager.schema import validate_tool_payload
from fleet.environments.task_manager.seed import (
    START_MS,
    STEP_MS,
    TASK_MANAGER_ASSIGNMENTS,
    TASK_MANAGER_DEPENDENCIES,
    TASK_MANAGER_MILESTONES,
    TASK_MANAGER_PROJECTS,
    TASK_MANAGER_TASKS,
    TASK_MANAGER_USERS,
)


def seed_database(db_path: Path, snapshot_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE users (
                user_id      TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                email        TEXT NOT NULL,
                role         TEXT NOT NULL,
                team         TEXT NOT NULL,
                handle       TEXT NOT NULL
            );
            CREATE TABLE projects (
                project_id   TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                description  TEXT NOT NULL DEFAULT '',
                owner_id     TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                archived     INTEGER NOT NULL DEFAULT 0
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
                task_id      TEXT PRIMARY KEY,
                title        TEXT NOT NULL,
                description  TEXT NOT NULL,
                creator_id   TEXT NOT NULL,
                assignee_id  TEXT,
                status       TEXT NOT NULL,
                created_at_ms  INTEGER NOT NULL,
                updated_at_ms  INTEGER NOT NULL,
                project_id   TEXT REFERENCES projects(project_id),
                milestone_id TEXT REFERENCES milestones(milestone_id),
                due_at_ms    INTEGER,
                priority     TEXT NOT NULL DEFAULT 'MEDIUM',
                labels       TEXT NOT NULL DEFAULT '[]',
                deleted      INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE assignments (
                assignment_id TEXT PRIMARY KEY,
                task_id       TEXT NOT NULL,
                user_id       TEXT NOT NULL,
                assigned_by   TEXT NOT NULL,
                assigned_at_ms INTEGER NOT NULL
            );
            CREATE TABLE task_dependencies (
                dep_id              TEXT PRIMARY KEY,
                task_id             TEXT NOT NULL REFERENCES tasks(task_id),
                depends_on_task_id  TEXT NOT NULL REFERENCES tasks(task_id)
            );
            CREATE TABLE audit_events (
                event_id          TEXT PRIMARY KEY,
                task_id           TEXT NOT NULL,
                actor_id          TEXT NOT NULL,
                event_type        TEXT NOT NULL,
                before_json       TEXT NOT NULL,
                after_json        TEXT NOT NULL,
                virtual_timestamp INTEGER NOT NULL
            );
            """
        )

        connection.executemany(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
            TASK_MANAGER_USERS,
        )

        connection.executemany(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)",
            [
                (project_id, name, description, owner_id, START_MS + offset * STEP_MS, 1 if archived else 0)
                for project_id, name, description, owner_id, offset, archived in TASK_MANAGER_PROJECTS
            ],
        )

        connection.executemany(
            "INSERT INTO milestones VALUES (?, ?, ?, ?, ?, ?)",
            [
                (milestone_id, project_id, title, description, due_at_ms, START_MS + offset * STEP_MS)
                for milestone_id, project_id, title, description, due_at_ms, offset in TASK_MANAGER_MILESTONES
            ],
        )

        connection.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    task_id, title, description, creator_id, assignee_id, status,
                    START_MS + offset * STEP_MS,
                    START_MS + offset * STEP_MS,
                    project_id, milestone_id, due_at_ms, priority,
                    json.dumps(list(labels)),
                    1 if status in ("DELETED", "DUPLICATE") else 0,
                )
                for (
                    task_id, title, description, creator_id, assignee_id, status,
                    project_id, milestone_id, due_at_ms, priority, labels, offset,
                ) in TASK_MANAGER_TASKS
            ],
        )

        connection.executemany(
            "INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
            [
                (assignment_id, task_id, user_id, assigned_by, START_MS + offset * STEP_MS)
                for assignment_id, task_id, user_id, assigned_by, offset in TASK_MANAGER_ASSIGNMENTS
            ],
        )

        connection.executemany(
            "INSERT INTO task_dependencies VALUES (?, ?, ?)",
            TASK_MANAGER_DEPENDENCIES,
        )

        connection.commit()
        snapshot_path.write_text("\n".join(connection.iterdump()), encoding="utf-8")


def teardown_database(db_path: Path, snapshot_path: Path) -> None:
    remove_pycaches()
    if not snapshot_path.exists():
        seed_database(db_path, snapshot_path)
        return
    if db_path.exists():
        db_path.unlink()
    with connect(db_path) as connection:
        connection.executescript(snapshot_path.read_text(encoding="utf-8"))


def query_rows(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    rows = plain_query_rows(connection, sql, params)
    for row in rows:
        if "labels" in row and isinstance(row["labels"], str):
            try:
                row["labels"] = json.loads(row["labels"])
            except (json.JSONDecodeError, TypeError):
                row["labels"] = []
    return rows


def export_state(db_path: Path) -> dict[str, Any]:
    with connect(db_path) as connection:
        return {
            "users":        query_rows(connection, "SELECT * FROM users ORDER BY user_id"),
            "projects":     query_rows(connection, "SELECT * FROM projects ORDER BY project_id"),
            "milestones":   query_rows(connection, "SELECT * FROM milestones ORDER BY milestone_id"),
            "tasks":        query_rows(connection, "SELECT * FROM tasks ORDER BY task_id"),
            "assignments":  query_rows(connection, "SELECT * FROM assignments ORDER BY assignment_id"),
            "dependencies": query_rows(connection, "SELECT * FROM task_dependencies ORDER BY dep_id"),
            "audit_events": query_rows(connection, "SELECT * FROM audit_events ORDER BY event_id"),
        }


# ---------------------------------------------------------------------------
# Task CRUD
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
    params:  list[Any] = []
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
        tasks = query_rows(connection, f"SELECT * FROM tasks {where} ORDER BY title", tuple(params))
    return {"tasks": tasks}


def get_task(db_path: Path, task_id: str) -> dict[str, Any]:
    with connect(db_path) as connection:
        task = connection.execute(
            "SELECT * FROM tasks WHERE task_id = ? AND deleted = 0",
            (task_id,),
        ).fetchone()
        if task is None:
            raise ToolError("task_not_found", "not_found", "Task was not found.")
        row = dict(task)
        if isinstance(row.get("labels"), str):
            row["labels"] = json.loads(row["labels"] or "[]")
        depends_on = [
            r["depends_on_task_id"]
            for r in query_rows(
                connection,
                "SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ?",
                (task_id,),
            )
        ]
        required_by = [
            r["task_id"]
            for r in query_rows(
                connection,
                "SELECT task_id FROM task_dependencies WHERE depends_on_task_id = ?",
                (task_id,),
            )
        ]
    return {"id": task_id, "task": row, "depends_on": depends_on, "required_by": required_by}


def create_task(
    db_path: Path,
    title: str,
    description: str = "",
    actor_id: str = "U001",
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
    assignee_id = assignee_id.strip()
    labels_json = json.dumps(labels or [])
    with connect(db_path) as connection:
        actor = connection.execute("SELECT * FROM users WHERE user_id = ?", (actor_id,)).fetchone()
        if actor is None:
            raise ToolError("user_not_found", "permission_denied", "Acting user was not found.")
        if assignee_id:
            assignee = connection.execute("SELECT * FROM users WHERE user_id = ?", (assignee_id,)).fetchone()
            if assignee is None:
                raise ToolError("user_not_found", "not_found", "Assignee user was not found.")
        task_count = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        task_id = task_id.strip() or f"TASK{task_count + 1:03d}"
        if connection.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone():
            raise ToolError("duplicate_task", "conflict", "Task id already exists.")
        now_ms = START_MS + (task_count + 10) * STEP_MS
        deleted = 1 if status in ("DELETED", "DUPLICATE") else 0
        connection.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id, title, description.strip(), actor_id, assignee_id or None,
                status, now_ms, now_ms,
                project_id, milestone_id, due_at_ms, priority, labels_json, deleted,
            ),
        )
        if assignee_id:
            assignment_count = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
            connection.execute(
                "INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
                (f"ASSIGN{assignment_count + 1:03d}", task_id, assignee_id, actor_id, now_ms),
            )
        audit = insert_audit(
            connection, task_id, actor_id, "task_created", {},
            {"status": status, "title": title, "assignee_id": assignee_id},
        )
        connection.commit()
        task_row = dict(connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone())
        task_row["labels"] = json.loads(task_row.get("labels") or "[]")
    return {"id": task_id, "task": task_row, "audit_event": audit}


def update_task(
    db_path: Path,
    task_id: str,
    actor_id: str = "U001",
    title: str | None = None,
    description: str | None = None,
    assignee_id: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    project_id: str | None = None,
    milestone_id: str | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    with connect(db_path) as connection:
        task  = connection.execute("SELECT * FROM tasks WHERE task_id = ? AND deleted = 0", (task_id,)).fetchone()
        actor = connection.execute("SELECT * FROM users WHERE user_id = ?", (actor_id,)).fetchone()
        if task is None:
            raise ToolError("task_not_found", "not_found", "Task was not found.")
        if actor is None:
            raise ToolError("user_not_found", "permission_denied", "Acting user was not found.")

        updates: dict[str, Any] = {}
        before: dict[str, str] = {}
        after:  dict[str, str] = {}

        if title is not None:
            title = title.strip()
            if not title:
                raise ToolError("invalid_arguments", "validation_error", "Task title is required.")
            updates["title"] = title
        if description is not None:
            updates["description"] = description.strip()
        if assignee_id is not None:
            assignee_id = assignee_id.strip()
            if assignee_id:
                if connection.execute("SELECT 1 FROM users WHERE user_id = ?", (assignee_id,)).fetchone() is None:
                    raise ToolError("user_not_found", "not_found", "Assignee user was not found.")
            updates["assignee_id"] = assignee_id or None
        if status is not None:
            updates["status"] = normalize_status(status)
        if priority is not None:
            p = priority.upper()
            if p not in PRIORITY_VALUES:
                raise ToolError("invalid_arguments", "validation_error", "Unknown priority value.")
            updates["priority"] = p

        if project_id is not None:
            if connection.execute("SELECT 1 FROM projects WHERE project_id = ?", (project_id,)).fetchone() is None:
                raise ToolError("project_not_found", "not_found", "Project was not found.")
            updates["project_id"] = project_id
        if milestone_id is not None:
            if milestone_id:
                ms = connection.execute("SELECT * FROM milestones WHERE milestone_id = ?", (milestone_id,)).fetchone()
                if ms is None:
                    raise ToolError("milestone_not_found", "not_found", "Milestone was not found.")
            updates["milestone_id"] = milestone_id or None

        if labels is not None:
            updates["labels"] = json.dumps(labels)

        metadata_fields = {"title", "description", "assignee_id", "priority", "project_id", "milestone_id", "labels"}
        metadata_changed = any(f in updates and task[f] != updates[f] for f in metadata_fields)
        status_changed   = "status" in updates and task["status"] != updates["status"]

        if metadata_changed and actor["role"] != "admin" and task["creator_id"] != actor_id:
            raise ToolError("permission_denied", "permission_denied", "Only creators or admins can update task metadata.")
        is_assignee = connection.execute(
            "SELECT 1 FROM assignments WHERE task_id = ? AND user_id = ?", (task_id, actor_id),
        ).fetchone()
        if status_changed and actor["role"] != "admin" and task["creator_id"] != actor_id and not is_assignee:
            raise ToolError("permission_denied", "permission_denied", "User cannot update this task status.")
        if "status" in updates:
            new_status = updates["status"]
            if new_status not in VALID_STATUS_TRANSITIONS:
                raise ToolError("invalid_arguments", "validation_error", "Unknown task status.")
            if status_changed and new_status not in VALID_STATUS_TRANSITIONS[task["status"]]:
                raise ToolError("invalid_status_transition", "state_machine_error", f"Cannot transition task from {task['status']} to {new_status}.")
        if not metadata_changed and not status_changed:
            row = dict(task)
            row["labels"] = json.loads(row.get("labels") or "[]")
            return {"id": task_id, "task": row, "noop": True}

        for f, value in updates.items():
            if task[f] != value:
                before[f] = "" if task[f] is None else str(task[f])
                after[f]  = "" if value is None else str(value)

        now_ms = START_MS + 20 * STEP_MS
        set_parts = [f"{f} = ?" for f in updates]
        params: list[Any] = list(updates.values())
        set_parts.append("updated_at_ms = ?")
        params.append(now_ms)
        if updates.get("status") in ("DELETED", "DUPLICATE"):
            set_parts.append("deleted = ?")
            params.append(1)
        params.append(task_id)
        connection.execute(f"UPDATE tasks SET {', '.join(set_parts)} WHERE task_id = ?", tuple(params))

        if updates.get("assignee_id"):
            if not connection.execute(
                "SELECT 1 FROM assignments WHERE task_id = ? AND user_id = ?",
                (task_id, updates["assignee_id"]),
            ).fetchone():
                assignment_count = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
                connection.execute(
                    "INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
                    (f"ASSIGN{assignment_count + 1:03d}", task_id, updates["assignee_id"], actor_id, now_ms),
                )

        event_type = "status_changed" if set(updates) == {"status"} else "task_updated"
        audit = insert_audit(connection, task_id, actor_id, event_type, before, after)
        connection.commit()
        updated_row = dict(connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone())
        updated_row["labels"] = json.loads(updated_row.get("labels") or "[]")
    return {"id": task_id, "task": updated_row, "audit_event": audit}


def delete_task(db_path: Path, task_id: str, actor_id: str = "U001") -> dict[str, Any]:
    return update_task(db_path, task_id=task_id, actor_id=actor_id, status="DELETED")


def archive_task(db_path: Path, task_id: str, actor_id: str = "U001") -> dict[str, Any]:
    return update_task(db_path, task_id=task_id, actor_id=actor_id, status="ARCHIVED")


def mark_task_duplicate(db_path: Path, task_id: str, original_task_id: str, actor_id: str = "U001") -> dict[str, Any]:
    with connect(db_path) as connection:
        task     = connection.execute("SELECT * FROM tasks WHERE task_id = ? AND deleted = 0", (task_id,)).fetchone()
        actor    = connection.execute("SELECT * FROM users WHERE user_id = ?", (actor_id,)).fetchone()
        original = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (original_task_id,)).fetchone()
        if task is None:
            raise ToolError("task_not_found", "not_found", "Task was not found.")
        if actor is None:
            raise ToolError("user_not_found", "permission_denied", "Acting user was not found.")
        if original is None:
            raise ToolError("task_not_found", "not_found", "Original task was not found.")
        if task_id == original_task_id:
            raise ToolError("invalid_arguments", "validation_error", "A task cannot be a duplicate of itself.")
        if actor["role"] != "admin" and task["creator_id"] != actor_id:
            raise ToolError("permission_denied", "permission_denied", "Only creators or admins can mark tasks as duplicates.")
        now_ms = START_MS + 20 * STEP_MS
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
        updated_row = dict(connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone())
        updated_row["labels"] = json.loads(updated_row.get("labels") or "[]")
    return {"id": task_id, "task": updated_row, "audit_event": audit, "original_task_id": original_task_id}


def move_task_to_project(
    db_path: Path,
    task_id: str,
    project_id: str,
    milestone_id: str | None = None,
    actor_id: str = "U001",
) -> dict[str, Any]:
    with connect(db_path) as connection:
        task    = connection.execute("SELECT * FROM tasks WHERE task_id = ? AND deleted = 0", (task_id,)).fetchone()
        actor   = connection.execute("SELECT * FROM users WHERE user_id = ?", (actor_id,)).fetchone()
        project = connection.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        if task is None:
            raise ToolError("task_not_found", "not_found", "Task was not found.")
        if actor is None:
            raise ToolError("user_not_found", "permission_denied", "Acting user was not found.")
        if project is None:
            raise ToolError("project_not_found", "not_found", "Project was not found.")
        if actor["role"] != "admin" and task["creator_id"] != actor_id:
            raise ToolError("permission_denied", "permission_denied", "Only creators or admins can move tasks.")
        if milestone_id:
            ms = connection.execute("SELECT * FROM milestones WHERE milestone_id = ?", (milestone_id,)).fetchone()
            if ms is None:
                raise ToolError("milestone_not_found", "not_found", "Milestone was not found.")
            if ms["project_id"] != project_id:
                raise ToolError("invalid_arguments", "validation_error", "Milestone does not belong to the target project.")
        now_ms = START_MS + 20 * STEP_MS
        connection.execute(
            "UPDATE tasks SET project_id = ?, milestone_id = ?, updated_at_ms = ? WHERE task_id = ?",
            (project_id, milestone_id, now_ms, task_id),
        )
        before = {"project_id": task["project_id"] or "", "milestone_id": task["milestone_id"] or ""}
        after  = {"project_id": project_id, "milestone_id": milestone_id or ""}
        audit  = insert_audit(connection, task_id, actor_id, "task_moved", before, after)
        connection.commit()
        updated_row = dict(connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone())
        updated_row["labels"] = json.loads(updated_row.get("labels") or "[]")
    return {"id": task_id, "task": updated_row, "audit_event": audit}


def save_snapshot(db_path: Path, snapshot_path: Path) -> None:
    with connect(db_path) as connection:
        snapshot_path.write_text("\n".join(connection.iterdump()), encoding="utf-8")


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------

def create_project(
    db_path: Path,
    name: str,
    description: str = "",
    actor_id: str = "U001",
    project_id: str = "",
) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise ToolError("invalid_arguments", "validation_error", "Project name is required.")
    with connect(db_path) as connection:
        actor = connection.execute("SELECT * FROM users WHERE user_id = ?", (actor_id,)).fetchone()
        if actor is None:
            raise ToolError("user_not_found", "permission_denied", "Acting user was not found.")
        count = connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        project_id = project_id.strip() or f"P{count + 1:03d}"
        if connection.execute("SELECT 1 FROM projects WHERE project_id = ?", (project_id,)).fetchone():
            raise ToolError("duplicate_project", "conflict", "Project id already exists.")
        now_ms = START_MS + (count + 50) * STEP_MS
        connection.execute(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, name, description.strip(), actor_id, now_ms, 0),
        )
        connection.commit()
        project = dict(connection.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone())
    return {"id": project_id, "project": project}


def list_projects(db_path: Path, include_archived: bool = False) -> dict[str, Any]:
    where = "" if include_archived else "WHERE archived = 0"
    with connect(db_path) as connection:
        projects = query_rows(connection, f"SELECT * FROM projects {where} ORDER BY name")
    return {"projects": projects}


def get_project(db_path: Path, project_id: str) -> dict[str, Any]:
    with connect(db_path) as connection:
        project = connection.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        if project is None:
            raise ToolError("project_not_found", "not_found", "Project was not found.")
        tasks = query_rows(
            connection,
            "SELECT * FROM tasks WHERE project_id = ? AND deleted = 0 AND status != 'ARCHIVED' ORDER BY title",
            (project_id,),
        )
        milestones = query_rows(
            connection,
            "SELECT * FROM milestones WHERE project_id = ? ORDER BY title",
            (project_id,),
        )
    return {"id": project_id, "project": dict(project), "tasks": tasks, "milestones": milestones}


# ---------------------------------------------------------------------------
# Milestone CRUD
# ---------------------------------------------------------------------------

def create_milestone(
    db_path: Path,
    project_id: str,
    title: str,
    description: str = "",
    due_at_ms: int | None = None,
    actor_id: str = "U001",
    milestone_id: str = "",
) -> dict[str, Any]:
    title = title.strip()
    if not title:
        raise ToolError("invalid_arguments", "validation_error", "Milestone title is required.")
    with connect(db_path) as connection:
        if connection.execute("SELECT 1 FROM projects WHERE project_id = ?", (project_id,)).fetchone() is None:
            raise ToolError("project_not_found", "not_found", "Project was not found.")
        count = connection.execute("SELECT COUNT(*) FROM milestones").fetchone()[0]
        milestone_id = milestone_id.strip() or f"M{count + 1:03d}"
        if connection.execute("SELECT 1 FROM milestones WHERE milestone_id = ?", (milestone_id,)).fetchone():
            raise ToolError("duplicate_milestone", "conflict", "Milestone id already exists.")
        now_ms = START_MS + (count + 60) * STEP_MS
        connection.execute(
            "INSERT INTO milestones VALUES (?, ?, ?, ?, ?, ?)",
            (milestone_id, project_id, title, description.strip(), due_at_ms, now_ms),
        )
        connection.commit()
        milestone = dict(connection.execute("SELECT * FROM milestones WHERE milestone_id = ?", (milestone_id,)).fetchone())
    return {"id": milestone_id, "milestone": milestone}


# ---------------------------------------------------------------------------
# Dependency CRUD
# ---------------------------------------------------------------------------

def link_tasks(db_path: Path, task_id: str, depends_on_task_id: str) -> dict[str, Any]:
    with connect(db_path) as connection:
        for tid in (task_id, depends_on_task_id):
            if connection.execute("SELECT 1 FROM tasks WHERE task_id = ? AND deleted = 0", (tid,)).fetchone() is None:
                raise ToolError("task_not_found", "not_found", f"Task {tid} was not found.")
        existing = connection.execute(
            "SELECT * FROM task_dependencies WHERE task_id = ? AND depends_on_task_id = ?",
            (task_id, depends_on_task_id),
        ).fetchone()
        if existing:
            return {"id": existing["dep_id"], "dependency": dict(existing), "noop": True}
        count = connection.execute("SELECT COUNT(*) FROM task_dependencies").fetchone()[0]
        dep_id = f"DEP{count + 1:03d}"
        connection.execute(
            "INSERT INTO task_dependencies VALUES (?, ?, ?)",
            (dep_id, task_id, depends_on_task_id),
        )
        connection.commit()
    return {"id": dep_id, "dependency": {"dep_id": dep_id, "task_id": task_id, "depends_on_task_id": depends_on_task_id}}


def unlink_tasks(db_path: Path, task_id: str, depends_on_task_id: str) -> dict[str, Any]:
    with connect(db_path) as connection:
        dep = connection.execute(
            "SELECT dep_id FROM task_dependencies WHERE task_id = ? AND depends_on_task_id = ?",
            (task_id, depends_on_task_id),
        ).fetchone()
        if dep is None:
            raise ToolError("dependency_not_found", "not_found", "Dependency was not found.")
        connection.execute("DELETE FROM task_dependencies WHERE dep_id = ?", (dep["dep_id"],))
        connection.commit()
    return {"removed": True, "task_id": task_id, "depends_on_task_id": depends_on_task_id}


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------

def insert_audit(
    connection: sqlite3.Connection,
    task_id: str,
    actor_id: str,
    event_type: str,
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, Any]:
    event_count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    event_id = f"AUDIT{event_count + 1:03d}"
    virtual_timestamp = START_MS + (event_count + 30) * STEP_MS
    connection.execute(
        "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (event_id, task_id, actor_id, event_type,
         json.dumps(before, sort_keys=True),
         json.dumps(after,  sort_keys=True),
         virtual_timestamp),
    )
    return {
        "event_id": event_id, "task_id": task_id, "actor_id": actor_id,
        "event_type": event_type, "before": before, "after": after,
        "virtual_timestamp": virtual_timestamp,
    }


TOOL_NAMES: tuple[str, ...] = (
    "list_tasks", "get_task", "create_task", "update_task", "delete_task",
    "archive_task", "mark_task_duplicate", "create_project", "list_projects",
    "get_project", "create_milestone", "move_task_to_project", "link_tasks",
    "unlink_tasks",
)


def _parse_labels(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [l.strip() for l in value.split(",") if l.strip()]
    if isinstance(value, list):
        return [str(l) for l in value]
    return None


def execute_tool(db_path: Path, tool_name: str, input_payload: dict[str, Any], actor_id: str = "U001") -> dict[str, Any]:
    validation_error = validate_tool_payload(tool_name, input_payload)
    if validation_error:
        raise ToolError("invalid_arguments", "validation_error", f"invalid_arguments: {validation_error}")
    name = tool_name

    if name == "list_tasks":
        return list_tasks(
            db_path,
            include_deleted=bool(input_payload.get("include_deleted", False)),
            include_archived=bool(input_payload.get("include_archived", False)),
            project_id=input_payload.get("project_id"),
            status_filter=input_payload.get("status"),
            priority_filter=input_payload.get("priority"),
            milestone_filter=input_payload.get("milestone_id"),
            assignee_filter=input_payload.get("assignee"),
        )
    if name == "get_task":
        return get_task(db_path, str(input_payload.get("task_id", "")))
    if name == "create_task":
        raw_labels = _parse_labels(input_payload.get("labels"))
        assignee = input_payload.get("assignee")
        return create_task(
            db_path,
            title=str(input_payload.get("title", "")),
            description=str(input_payload.get("description", "")),
            actor_id=actor_id,
            task_id=input_payload.get("task_id") or "",
            assignee_id=assignee or "",
            status=input_payload.get("status") or "PENDING",
            project_id=input_payload.get("project_id"),
            milestone_id=input_payload.get("milestone_id"),
            due_at_ms=input_payload.get("due_at_ms"),
            priority=input_payload.get("priority") or "MEDIUM",
            labels=raw_labels,
        )
    if name == "update_task":
        labels = input_payload.get("labels")
        raw_labels = _parse_labels(labels) if isinstance(labels, (str, list)) else labels
        assignee = input_payload.get("assignee")
        return update_task(
            db_path,
            task_id=str(input_payload.get("task_id", "")),
            actor_id=actor_id,
            title=input_payload.get("title"),
            description=input_payload.get("description"),
            assignee_id=assignee,
            status=input_payload.get("status"),
            priority=input_payload.get("priority"),
            project_id=input_payload.get("project_id"),
            milestone_id=input_payload.get("milestone_id"),
            labels=raw_labels,
        )
    if name == "delete_task":
        return delete_task(db_path, str(input_payload.get("task_id", "")), actor_id)
    if name == "archive_task":
        return archive_task(db_path, str(input_payload.get("task_id", "")), actor_id)
    if name == "mark_task_duplicate":
        return mark_task_duplicate(
            db_path,
            task_id=str(input_payload.get("task_id", "")),
            original_task_id=str(input_payload.get("original_task_id", "")),
            actor_id=actor_id,
        )
    if name == "create_project":
        return create_project(
            db_path,
            name=str(input_payload.get("name", "")),
            description=str(input_payload.get("description", "")),
            actor_id=actor_id,
            project_id=input_payload.get("project_id") or "",
        )
    if name == "list_projects":
        return list_projects(db_path, bool(input_payload.get("include_archived", False)))
    if name == "get_project":
        return get_project(db_path, str(input_payload.get("project_id", "")))
    if name == "create_milestone":
        return create_milestone(
            db_path,
            project_id=str(input_payload.get("project_id", "")),
            title=str(input_payload.get("title", "")),
            description=str(input_payload.get("description", "")),
            due_at_ms=input_payload.get("due_at_ms"),
            actor_id=actor_id,
            milestone_id=input_payload.get("milestone_id") or "",
        )
    if name == "move_task_to_project":
        return move_task_to_project(
            db_path,
            task_id=str(input_payload.get("task_id", "")),
            project_id=str(input_payload.get("project_id", "")),
            milestone_id=input_payload.get("milestone_id"),
            actor_id=actor_id,
        )
    if name == "link_tasks":
        return link_tasks(
            db_path,
            str(input_payload.get("task_id", "")),
            str(input_payload.get("depends_on_task_id", "")),
        )
    if name == "unlink_tasks":
        return unlink_tasks(
            db_path,
            str(input_payload.get("task_id", "")),
            str(input_payload.get("depends_on_task_id", "")),
        )
    raise RuntimeError(f"Unknown tool: {tool_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "seed", "teardown", "state", "save-snapshot",
            "list-tasks", "get-task", "create-task", "update-task", "delete-task",
            "archive-task", "mark-task-duplicate", "move-task-to-project",
            "create-project", "list-projects", "get-project",
            "create-milestone",
            "link-tasks", "unlink-tasks", "execute_tool"
        ],
    )
    parser.add_argument("--tool-name", default="")
    parser.add_argument("--input-payload", default="")
    parser.add_argument("tool_name_pos", nargs="?", default="")
    parser.add_argument("input_payload_pos", nargs="?", default="")
    parser.add_argument("--db",              default="/app/task_manager.db")
    parser.add_argument("--snapshot",        default="/app/task_manager_seed_snapshot.sql")
    parser.add_argument("--actor-id",        default="U001")
    parser.add_argument("--include-deleted",  action="store_true")
    parser.add_argument("--include-archived", action="store_true")
    parser.add_argument("--title",           default="")
    parser.add_argument("--description",     default="")
    parser.add_argument("--assignee",        default="")
    parser.add_argument("--task-id",         default="")
    parser.add_argument("--user-id",         default="")
    parser.add_argument("--status",          default="")
    parser.add_argument("--project-id",      default="")
    parser.add_argument("--milestone-id",    default="")
    parser.add_argument("--priority",        default="")
    parser.add_argument("--due-at-ms",       type=int, default=None)
    parser.add_argument("--depends-on",      default="")
    parser.add_argument("--original-task-id", default="")
    parser.add_argument("--labels",          default="")
    args = parser.parse_args()

    db_path       = Path(args.db)
    snapshot_path = Path(args.snapshot)

    if args.command == "seed":
        seed_database(db_path, snapshot_path)
    elif args.command == "teardown":
        teardown_database(db_path, snapshot_path)
    elif args.command == "state":
        print(json.dumps(export_state(db_path), sort_keys=True, indent=2))
    elif args.command == "save-snapshot":
        save_snapshot(db_path, snapshot_path)
        print(json.dumps({"saved": str(snapshot_path)}))
    elif args.command == "list-tasks":
        print(json.dumps(list_tasks(
            db_path,
            args.include_deleted,
            args.include_archived,
            project_id=args.project_id or None,
            status_filter=args.status or None,
            priority_filter=args.priority or None,
            milestone_filter=args.milestone_id or None,
            assignee_filter=args.assignee or args.user_id or None,
        ), sort_keys=True, indent=2))
    elif args.command == "get-task":
        print(json.dumps(get_task(db_path, args.task_id), sort_keys=True, indent=2))
    elif args.command == "create-task":
        raw_labels = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels else []
        print(json.dumps(create_task(
            db_path, args.title, args.description, args.actor_id, args.task_id,
            args.assignee or args.user_id, args.status or "PENDING",
            project_id=args.project_id or None,
            milestone_id=args.milestone_id or None,
            due_at_ms=args.due_at_ms,
            priority=args.priority or "MEDIUM",
            labels=raw_labels or None,
        ), sort_keys=True, indent=2))
    elif args.command == "update-task":
        raw_labels = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels else None
        print(json.dumps(update_task(
            db_path, task_id=args.task_id, actor_id=args.actor_id,
            title=args.title or None,
            description=args.description or None,
            assignee_id=args.assignee or args.user_id or None,
            status=args.status or None,
            priority=args.priority or None,
            project_id=args.project_id or None,
            milestone_id=args.milestone_id or None,
            labels=raw_labels,
        ), sort_keys=True, indent=2))
    elif args.command == "delete-task":
        print(json.dumps(delete_task(db_path, args.task_id, args.actor_id), sort_keys=True, indent=2))
    elif args.command == "archive-task":
        print(json.dumps(archive_task(db_path, args.task_id, args.actor_id), sort_keys=True, indent=2))
    elif args.command == "mark-task-duplicate":
        print(json.dumps(mark_task_duplicate(db_path, args.task_id, args.original_task_id, args.actor_id), sort_keys=True, indent=2))
    elif args.command == "move-task-to-project":
        print(json.dumps(move_task_to_project(
            db_path, args.task_id, args.project_id,
            milestone_id=args.milestone_id or None,
            actor_id=args.actor_id,
        ), sort_keys=True, indent=2))
    elif args.command == "create-project":
        print(json.dumps(create_project(
            db_path, args.title, args.description, args.actor_id,
            project_id=args.project_id or "",
        ), sort_keys=True, indent=2))
    elif args.command == "list-projects":
        print(json.dumps(list_projects(db_path, args.include_archived), sort_keys=True, indent=2))
    elif args.command == "get-project":
        print(json.dumps(get_project(db_path, args.project_id), sort_keys=True, indent=2))
    elif args.command == "create-milestone":
        print(json.dumps(create_milestone(
            db_path, args.project_id, args.title, args.description,
            due_at_ms=args.due_at_ms, actor_id=args.actor_id,
            milestone_id=args.milestone_id or "",
        ), sort_keys=True, indent=2))
    elif args.command == "link-tasks":
        print(json.dumps(link_tasks(db_path, args.task_id, args.depends_on), sort_keys=True, indent=2))
    elif args.command == "unlink-tasks":
        print(json.dumps(unlink_tasks(db_path, args.task_id, args.depends_on), sort_keys=True, indent=2))
    elif args.command == "execute_tool":
        tool_name = args.tool_name or args.tool_name_pos
        payload_str = args.input_payload or args.input_payload_pos or "{}"
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON input payload: {payload_str}") from exc
        print(json.dumps(execute_tool(db_path, tool_name, payload, args.actor_id), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
