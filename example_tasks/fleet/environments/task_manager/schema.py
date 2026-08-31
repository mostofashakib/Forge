"""Tool schema for agents operating the Task Manager service."""

from __future__ import annotations

from typing import Any

ToolSchema = dict[str, Any]

_TASK_STATUS_ENUM = ["PENDING", "IN_PROGRESS", "BLOCKED", "COMPLETED", "CANCELLED", "DELETED", "ARCHIVED", "DUPLICATE"]
_PRIORITY_ENUM = ["LOW", "MEDIUM", "HIGH", "URGENT"]

TASK_MANAGER_TOOL_SCHEMA: list[ToolSchema] = [
    # ------------------------------------------------------------------
    # Core task CRUD
    # ------------------------------------------------------------------
    {
        "name": "list_tasks",
        "description": "List task records visible to the acting user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_deleted":  {"type": "boolean", "description": "Include soft-deleted (DELETED, DUPLICATE) tasks.", "default": False},
                "include_archived": {"type": "boolean", "description": "Include ARCHIVED tasks.", "default": False},
                "project_id":       {"type": "string",  "description": "Filter by project id."},
                "milestone_id":     {"type": "string",  "description": "Filter by milestone id."},
                "status":           {"type": "string",  "enum": _TASK_STATUS_ENUM, "description": "Filter by status."},
                "priority":         {"type": "string",  "enum": _PRIORITY_ENUM,   "description": "Filter by priority."},
                "assignee":         {"type": "string",  "description": "Filter by assignee user id."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_task",
        "description": "Fetch one task by stable task id, including archived tasks. Response includes depends_on (upstream task ids this task must wait for) and required_by (downstream task ids waiting on this task). Both lists reflect current dependency state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Stable task id."},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_task",
        "description": "Create a task with optional stable id, assignee, project, milestone, priority, and labels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":      {"type": "string",  "description": "Optional stable task id. Generated if omitted."},
                "title":        {"type": "string",  "description": "Task title."},
                "description":  {"type": "string",  "description": "Task description.", "default": ""},
                "assignee":     {"type": "string",  "description": "Optional assignee user id."},
                "status":       {"type": "string",  "enum": _TASK_STATUS_ENUM, "description": "Initial status. Defaults to PENDING.", "default": "PENDING"},
                "project_id":   {"type": "string",  "description": "Optional project id."},
                "milestone_id": {"type": "string",  "description": "Optional milestone id."},
                "due_at_ms":    {"type": "integer", "description": "Optional due date as Unix timestamp in milliseconds."},
                "priority":     {"type": "string",  "enum": _PRIORITY_ENUM, "description": "Task priority. Defaults to MEDIUM.", "default": "MEDIUM"},
                "labels":       {"type": "array",   "items": {"type": "string"}, "description": "Optional label tags."},
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_task",
        "description": "Update task title, description, assignee, status, project, milestone, priority, or labels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":      {"type": "string",  "description": "Task id."},
                "title":        {"type": "string",  "description": "New task title."},
                "description":  {"type": "string",  "description": "New task description."},
                "assignee":     {"type": "string",  "description": "New assignee user id."},
                "status":       {"type": "string",  "enum": _TASK_STATUS_ENUM, "description": "New task status."},
                "project_id":   {"type": "string",  "description": "Move to this project (null to unset)."},
                "milestone_id": {"type": "string",  "description": "Set milestone (null to unset)."},
                "due_at_ms":    {"type": "integer", "description": "New due date as Unix timestamp in milliseconds."},
                "priority":     {"type": "string",  "enum": _PRIORITY_ENUM, "description": "New task priority."},
                "labels":       {"type": "array",   "items": {"type": "string"}, "description": "Replace label tags."},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "delete_task",
        "description": "Soft-delete an active task by setting status to DELETED.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task id."},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
    # ------------------------------------------------------------------
    # Archival and deduplication
    # ------------------------------------------------------------------
    {
        "name": "archive_task",
        "description": "Archive a task, marking it as no longer active but preserving its full history. "
                       "Distinct from DELETED: archived tasks are retained and can be listed with include_archived=true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task id."},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mark_task_duplicate",
        "description": "Mark a task as a duplicate of another task. The duplicate is hidden from default listings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":          {"type": "string", "description": "The task to mark as duplicate."},
                "original_task_id": {"type": "string", "description": "The canonical task this duplicates."},
            },
            "required": ["task_id", "original_task_id"],
            "additionalProperties": False,
        },
    },
    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------
    {
        "name": "create_project",
        "description": "Create a new project container for tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id":  {"type": "string", "description": "Optional stable project id. Generated if omitted."},
                "name":        {"type": "string", "description": "Project name."},
                "description": {"type": "string", "description": "Project description.", "default": ""},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_projects",
        "description": "List all projects.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_archived": {"type": "boolean", "description": "Include archived projects.", "default": False},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_project",
        "description": "Get a project with its tasks and milestones.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project id."},
            },
            "required": ["project_id"],
            "additionalProperties": False,
        },
    },
    # ------------------------------------------------------------------
    # Milestones
    # ------------------------------------------------------------------
    {
        "name": "create_milestone",
        "description": "Create a milestone checkpoint within a project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "milestone_id": {"type": "string",  "description": "Optional stable milestone id. Generated if omitted."},
                "project_id":   {"type": "string",  "description": "Project the milestone belongs to."},
                "title":        {"type": "string",  "description": "Milestone title."},
                "description":  {"type": "string",  "description": "Milestone description.", "default": ""},
                "due_at_ms":    {"type": "integer", "description": "Due date as Unix timestamp in milliseconds."},
            },
            "required": ["project_id", "title"],
            "additionalProperties": False,
        },
    },
    # ------------------------------------------------------------------
    # Task movement and linking
    # ------------------------------------------------------------------
    {
        "name": "move_task_to_project",
        "description": "Assign or move a task to a project, optionally setting a milestone within that project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":      {"type": "string", "description": "Task to move."},
                "project_id":   {"type": "string", "description": "Destination project id."},
                "milestone_id": {"type": "string", "description": "Optional milestone id within the destination project."},
            },
            "required": ["task_id", "project_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "link_tasks",
        "description": "Declare that one task depends on another: task_id cannot be considered complete until "
                       "depends_on_task_id is COMPLETED.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":          {"type": "string", "description": "The downstream (blocked) task."},
                "depends_on_task_id": {"type": "string", "description": "The upstream (blocking) task."},
            },
            "required": ["task_id", "depends_on_task_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "unlink_tasks",
        "description": "Remove an existing dependency between two tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":          {"type": "string", "description": "The downstream task."},
                "depends_on_task_id": {"type": "string", "description": "The upstream task to unlink."},
            },
            "required": ["task_id", "depends_on_task_id"],
            "additionalProperties": False,
        },
    },
]


def validate_tool_payload(tool_name: str, payload: Any) -> str | None:
    """Check a tool input payload against the declared schema; return an error
    message, or None when valid. Enforces the contract the schema advertises
    (`additionalProperties: False`, required keys, enums) so that a misspelled
    parameter fails loudly instead of being silently ignored. Enum values are
    matched case-insensitively to mirror status/priority normalization. None
    values and per-property types are left to the tools themselves."""
    tool = next((t for t in TASK_MANAGER_TOOL_SCHEMA if t["name"] == tool_name), None)
    if tool is None:
        valid_names = sorted(t["name"] for t in TASK_MANAGER_TOOL_SCHEMA)
        return f"Unknown tool: {tool_name}. Valid tools: {valid_names}"
    if not isinstance(payload, dict):
        return f"Input payload for {tool_name} must be a JSON object."
    schema = tool["input_schema"]
    properties = schema.get("properties", {})
    unknown = sorted(set(payload) - set(properties))
    if unknown:
        return (
            f"Unexpected parameter(s) {unknown} for tool {tool_name}. "
        )
    missing = sorted(k for k in schema.get("required", []) if k not in payload or payload[k] is None)
    if missing:
        return f"Missing required parameter(s) {missing} for tool {tool_name}."
    for key, spec in properties.items():
        value = payload.get(key)
        enum = spec.get("enum")
        if value is None or not enum:
            continue
        if str(value).upper() not in {str(option).upper() for option in enum}:
            return f"Invalid value {value!r} for parameter {key} of tool {tool_name}. Allowed values: {enum}"
    return None
