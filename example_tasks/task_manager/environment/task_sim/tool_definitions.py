"""Model-facing JSON-schema definitions for the task-tracker actions.

One list, generated into every surface: the MCP server Harbor registers, the
`tasks` CLI the oracle drives, and the RL environment's `tool_definitions()`.
A tool that exists in the world therefore cannot go missing from what the agent
is given, and a flag the CLI accepts cannot drift from a field the schema
declares -- a test asserts both.
"""

from __future__ import annotations

import copy
from typing import Any


def _tool(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TEXT = {"type": "string"}
BOOLEAN = {"type": "boolean"}
INTEGER = {"type": "integer"}
STRING_LIST = {"type": "array", "items": {"type": "string"}}

STATUS_VALUES = [
    "PENDING", "IN_PROGRESS", "BLOCKED", "COMPLETED",
    "CANCELLED", "DELETED", "ARCHIVED", "DUPLICATE",
]
PRIORITY_VALUES = ["LOW", "MEDIUM", "HIGH", "URGENT"]

STATUS = {"type": "string", "enum": STATUS_VALUES}
PRIORITY = {"type": "string", "enum": PRIORITY_VALUES}

TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    # -- reading ------------------------------------------------------------
    _tool(
        "list_tasks",
        "List task records visible to the acting user, newest filters first. "
        "Archived and closed tasks are hidden unless you ask for them. Every "
        "filter is optional and they combine.",
        {
            "include_deleted": BOOLEAN,
            "include_archived": BOOLEAN,
            "project_id": TEXT,
            "milestone_id": TEXT,
            "status": STATUS,
            "priority": PRIORITY,
            "assignee": TEXT,
        },
        [],
    ),
    _tool(
        "get_task",
        "Fetch one task by id, archived tasks included. The response carries "
        "depends_on (upstream tasks this one waits for) and required_by "
        "(downstream tasks waiting on it), both reflecting current state.",
        {"task_id": TEXT},
        ["task_id"],
    ),
    _tool("list_users", "List workspace users with their team and role.", {}, []),
    _tool(
        "list_projects",
        "List projects. Archived projects are hidden unless you ask for them.",
        {"include_archived": BOOLEAN},
        [],
    ),
    _tool(
        "get_project",
        "Get one project together with its open tasks and its milestones.",
        {"project_id": TEXT},
        ["project_id"],
    ),
    # -- writing ------------------------------------------------------------
    _tool(
        "create_task",
        "Create a task with an optional stable id, assignee, project, "
        "milestone, priority, and labels.",
        {
            "task_id": TEXT,
            "title": TEXT,
            "description": TEXT,
            "assignee": TEXT,
            "status": STATUS,
            "project_id": TEXT,
            "milestone_id": TEXT,
            "due_at_ms": INTEGER,
            "priority": PRIORITY,
            "labels": STRING_LIST,
        },
        ["title"],
    ),
    _tool(
        "update_task",
        "Update a task's title, description, assignee, status, project, "
        "milestone, priority, or labels. labels REPLACES the list, so read the "
        "task first if you mean to add one.",
        {
            "task_id": TEXT,
            "title": TEXT,
            "description": TEXT,
            "assignee": TEXT,
            "status": STATUS,
            "project_id": TEXT,
            "milestone_id": TEXT,
            "due_at_ms": INTEGER,
            "priority": PRIORITY,
            "labels": STRING_LIST,
        },
        ["task_id"],
    ),
    _tool(
        "delete_task",
        "Close a task as DELETED. Irreversible: a deleted task cannot return to "
        "any active status.",
        {"task_id": TEXT},
        ["task_id"],
    ),
    _tool(
        "archive_task",
        "Archive a task: no longer active, full history preserved. Distinct "
        "from DELETED, and listed again with include_archived.",
        {"task_id": TEXT},
        ["task_id"],
    ),
    _tool(
        "mark_task_duplicate",
        "Mark a task as a duplicate of another. The duplicate is hidden from "
        "default listings.",
        {"task_id": TEXT, "original_task_id": TEXT},
        ["task_id", "original_task_id"],
    ),
    _tool(
        "move_task_to_project",
        "Move a task to a project, optionally setting a milestone within it.",
        {"task_id": TEXT, "project_id": TEXT, "milestone_id": TEXT},
        ["task_id", "project_id"],
    ),
    _tool(
        "create_project",
        "Create a new project container for tasks.",
        {"project_id": TEXT, "name": TEXT, "description": TEXT},
        ["name"],
    ),
    _tool(
        "create_milestone",
        "Create a milestone checkpoint within a project.",
        {
            "milestone_id": TEXT,
            "project_id": TEXT,
            "title": TEXT,
            "description": TEXT,
            "due_at_ms": INTEGER,
        },
        ["project_id", "title"],
    ),
    _tool(
        "link_tasks",
        "Declare that one task depends on another: task_id cannot be complete "
        "until depends_on_task_id is COMPLETED.",
        {"task_id": TEXT, "depends_on_task_id": TEXT},
        ["task_id", "depends_on_task_id"],
    ),
    _tool(
        "unlink_tasks",
        "Remove an existing dependency between two tasks.",
        {"task_id": TEXT, "depends_on_task_id": TEXT},
        ["task_id", "depends_on_task_id"],
    ),
    # -- reporting ----------------------------------------------------------
    #
    # The graded answer has to live somewhere the world owns. Grading reads the
    # state export and the action log, never the agent's own trajectory, so a
    # final answer returned only to the harness would be an answer the verifier
    # is not allowed to look at. This tool is where the agent states its result,
    # and the world records it like any other write.
    _tool(
        "submit_handover_report",
        "Record the outcome of your work: the task ids you modified, and a "
        "one-line summary. Submitting again replaces the previous report, so "
        "the last one stands.",
        {"task_ids": STRING_LIST, "summary": TEXT},
        ["task_ids"],
    ),
)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return independent dictionaries safe for callers to modify."""
    return copy.deepcopy(list(TOOL_DEFINITIONS))


def tool_names() -> tuple[str, ...]:
    return tuple(tool["name"] for tool in TOOL_DEFINITIONS)


def validate_tool_payload(tool_name: str, payload: Any) -> str | None:
    """Check one tool input against the declared schema.

    Returns an error message, or None when valid. Enforces exactly what the
    schema advertises -- `additionalProperties: false`, required keys, enums --
    so a misspelled parameter fails loudly instead of being silently ignored.
    Enum values match case-insensitively, mirroring status/priority
    normalization; per-property types are left to the tools themselves.
    """
    tool = next((item for item in TOOL_DEFINITIONS if item["name"] == tool_name), None)
    if tool is None:
        return f"Unknown tool: {tool_name}. Valid tools: {sorted(tool_names())}"
    if not isinstance(payload, dict):
        return f"Input payload for {tool_name} must be a JSON object."
    schema = tool["input_schema"]
    properties = schema.get("properties", {})
    unknown = sorted(set(payload) - set(properties))
    if unknown:
        return f"Unexpected parameter(s) {unknown} for tool {tool_name}."
    missing = sorted(
        key for key in schema.get("required", [])
        if key not in payload or payload[key] is None
    )
    if missing:
        return f"Missing required parameter(s) {missing} for tool {tool_name}."
    for key, spec in properties.items():
        value = payload.get(key)
        enum = spec.get("enum")
        if value is None or not enum:
            continue
        if str(value).upper() not in {str(option).upper() for option in enum}:
            return (
                f"Invalid value {value!r} for parameter {key} of tool {tool_name}. "
                f"Allowed values: {enum}"
            )
    return None
