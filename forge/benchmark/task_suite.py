from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

Difficulty = int  # 1–5


@dataclass
class Task:
    name: str
    domain: str
    objective: str
    success_fn: Callable[[dict], bool]
    difficulty: Difficulty


class TaskSuite:
    def __init__(self) -> None:
        self._tasks: list[Task] = _build_registry()

    def domains(self) -> list[str]:
        return sorted({t.domain for t in self._tasks})

    def tasks_for(self, domain: str, depth: int) -> list[Task]:
        return [t for t in self._tasks if t.domain == domain and t.difficulty <= depth]

    def all_tasks(self, max_depth: int = 5) -> list[Task]:
        return [t for t in self._tasks if t.difficulty <= max_depth]


DOMAINS = ["email", "project_mgmt"]


def _build_registry() -> list[Task]:
    return [
        # ── Email domain ─────────────────────────────────────────────────────
        Task(
            name="email_read_star",
            domain="email",
            objective="Open the first unread email and star it.",
            success_fn=lambda s: bool(s.get("selected_email")) and bool(s.get("starred_count", 0) > 0),
            difficulty=1,
        ),
        Task(
            name="email_reply_label",
            domain="email",
            objective="Reply to the most recent email and apply the label 'Reviewed'.",
            success_fn=lambda s: bool(s.get("last_reply_sent")) and "Reviewed" in s.get("active_labels", []),
            difficulty=2,
        ),
        Task(
            name="email_search_bulk_archive",
            domain="email",
            objective="Search for emails from 'noreply@example.com' and archive all results.",
            success_fn=lambda s: len(s.get("search_results", [])) > 0 and bool(s.get("bulk_action_completed")),
            difficulty=3,
        ),
        Task(
            name="email_search_read_reply_label",
            domain="email",
            objective="Search for the invoice email, open it, reply with 'Received', then label it 'Finance'.",
            success_fn=lambda s: (
                bool(s.get("last_reply_sent"))
                and "Finance" in s.get("active_labels", [])
            ),
            difficulty=4,
        ),
        Task(
            name="email_conditional_filter_schedule_send",
            domain="email",
            objective=(
                "Find all emails with subject containing 'Urgent', draft a reply to each "
                "saying 'Acknowledged', and schedule them to send tomorrow at 9am."
            ),
            success_fn=lambda s: bool(s.get("scheduled_sends_count", 0) > 0),
            difficulty=5,
        ),
        # ── Project management domain ─────────────────────────────────────────
        Task(
            name="pm_view_mark_done",
            domain="project_mgmt",
            objective="View the task list and mark the first incomplete task as done.",
            success_fn=lambda s: bool(s.get("last_completed_task")),
            difficulty=1,
        ),
        Task(
            name="pm_create_assign",
            domain="project_mgmt",
            objective="Create a new task called 'Review PR' and assign it to 'alice'.",
            success_fn=lambda s: any(
                t.get("title") == "Review PR" and t.get("assignee") == "alice"
                for t in s.get("task_list", [])
            ),
            difficulty=2,
        ),
        Task(
            name="pm_filter_set_deadline",
            domain="project_mgmt",
            objective="Filter tasks by status 'In Progress' and set a deadline of next Friday on each.",
            success_fn=lambda s: bool(s.get("bulk_deadline_set")),
            difficulty=3,
        ),
        Task(
            name="pm_find_blocked_reassign_notify",
            domain="project_mgmt",
            objective=(
                "Find all tasks with status 'Blocked', reassign them to 'bob', "
                "and add a comment 'Unblocking in progress'."
            ),
            success_fn=lambda s: bool(s.get("reassigned_count", 0) > 0) and bool(s.get("last_comment")),
            difficulty=4,
        ),
        Task(
            name="pm_cross_project_dependency",
            domain="project_mgmt",
            objective=(
                "Identify tasks in 'Project Alpha' that are blocked by incomplete tasks in 'Project Beta', "
                "create a dependency link between them, and notify the assignees."
            ),
            success_fn=lambda s: bool(s.get("dependency_links_created", 0) > 0),
            difficulty=5,
        ),
    ]
