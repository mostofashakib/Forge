"""Per-task agent configuration, kept out of the agent classes.

A profile declares how to recognize an instruction family and the
answer-format contract the verifier will hold the agent to. Profiles must not
contain solution-path hints (which tools to call, in what order, or
task-specific IDs): those would make the eval measure instruction-following
instead of reasoning. Adding a task means adding a profile here, not editing
agent code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    """Matches when every `all_of` term and at least one `any_of` term (if
    given) appear in the lowercased instruction."""

    all_of: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()
    guidance: str = ""

    def matches(self, instruction: str) -> bool:
        normalized = instruction.lower()
        if not all(term in normalized for term in self.all_of):
            return False
        if self.any_of and not any(term in normalized for term in self.any_of):
            return False
        return True


def find_profile(profiles: tuple[TaskProfile, ...], instruction: str) -> TaskProfile | None:
    for profile in profiles:
        if profile.matches(instruction):
            return profile
    return None


SLACK_TASK_PROFILES: tuple[TaskProfile, ...] = (
    # Incident-reference lookup (slack_task_1 family): answer format only.
    TaskProfile(
        all_of=("alice",),
        any_of=("references", "referenced"),
        guidance=(
            "The final answer must be the referenced channel name including the leading #, "
            "for example #example-channel."
        ),
    ),
)

SLACK_DEFAULT_GUIDANCE = (
    "Perform the requested Slack operations using the appropriate mutation tools, "
    "then return the final answer as requested in the instructions."
)

TASK_MANAGER_DEFAULT_GUIDANCE = (
    "The initial state contains tasks, projects, milestones, and dependencies. "
    "Use list_tasks with project_id/priority/status filters to narrow results. "
    "Use get_project to see a project's milestones. "
    "Use uppercase statuses: PENDING, IN_PROGRESS, COMPLETED, CANCELLED, DELETED, ARCHIVED, DUPLICATE. "
    "Uppercase priorities: LOW, MEDIUM, HIGH, URGENT. "
    "Mutate only the entities the instruction requires."
)
