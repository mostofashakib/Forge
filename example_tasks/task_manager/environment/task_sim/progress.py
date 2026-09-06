"""World-state readings for two callers.

The Harbor verifier asks whether the episode happened in the world this task
specifies; the RL environment asks how far along it is. Neither question is the
reward -- that is the layered verifier's, in `verifiers.contracts.reassignment`.

The split in what each answers is deliberate. Shaping counts only what the world
can see: which events fired, and whether a report was filed at all. Whether the
reported set is *right* is the verifier's call, after the episode. A list of
five ids is a short string a policy could assert without reading anything, so
rewarding its correctness per step would put the shaped signal in direct
opposition to the graded one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from task_sim.seed import (
    REASSIGNED_TASKS,
    REASSIGNMENT_SCENARIO,
    TRACKER_TASKS,
)

#: What a rollout is scored against, in the order the work happens.
WORKSPACE_MILESTONES = tuple(event.event_id for event in REASSIGNMENT_SCENARIO.events)

#: Every seeded task id. A row that leaves this set without a tool having closed
#: it is the store losing data, not the agent doing something.
SEEDED_TASK_IDS = frozenset(row[0] for row in TRACKER_TASKS)

#: Closing or re-homing a task nobody asked about. Each is a real side effect
#: the state export records, and each costs the shaped signal something.
_SIDE_EFFECT_PER_ACTION = 0.05


class IntegrityError(RuntimeError):
    """The episode did not happen in the world this task specifies.

    Raising one zeroes the run *and* sets `valid=0`, which says the episode is
    not evidence about the agent at all. That makes the exception the entire
    record of why a run was thrown away -- so it carries its evidence as data,
    not only as a sentence.

    The subclasses are not degrees of one problem. A malformed export means the
    harness is broken; vanished tasks mean the store was damaged; a visibility
    mismatch means the scenario engine and its own ledger disagree; an unknown
    event means the ledger was rewritten. Each sends a different person to look
    at a different thing, and one shared class name with a formatted string made
    all four look alike in the reward file.

    Every subclass stays reachable as `IntegrityError`: the Harbor verifier
    imports exactly that name and catches by it, so a sibling that escaped this
    base would reach the reward file as an unhandled exception instead of a
    `valid=0` with a reason.
    """

    kind = "integrity_error"

    def evidence(self) -> dict[str, Any]:
        return {}

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": str(self), **self.evidence()}


class MissingSectionsError(IntegrityError):
    """The state export does not have the sections a grader reads.

    Nothing the agent can do produces this. It means the exporter, the socket or
    the collection step failed, and the run should be re-collected rather than
    interpreted.
    """

    kind = "missing_sections"

    def __init__(self, sections: Iterable[str]) -> None:
        self.sections = sorted(sections)
        super().__init__(f"workspace export is missing sections: {self.sections}")

    def evidence(self) -> dict[str, Any]:
        return {"sections": self.sections}


class DestroyedHistoryError(IntegrityError):
    """Seeded tasks vanished that no tool can remove.

    Closing a task as DELETED is a real, vetoed side effect and is not this: a
    deleted task is still a row. This is the store losing rows, which says
    nothing about the agent.
    """

    kind = "destroyed_history"

    def __init__(self, task_ids: Iterable[str]) -> None:
        self.task_ids = sorted(task_ids)
        shown = self.task_ids[:5]
        suffix = (
            f" (+{len(self.task_ids) - len(shown)} more)"
            if len(self.task_ids) > len(shown) else ""
        )
        super().__init__(f"seeded tasks vanished from the workspace: {shown}{suffix}")

    def evidence(self) -> dict[str, Any]:
        return {"task_ids": self.task_ids}


class LatentVisibilityError(IntegrityError):
    """Released latent rows do not match the event ledger.

    Which direction it broke in matters: `unexpected` means content appeared
    without the event that releases it -- the agent was shown the future --
    while `missing` means a fired event released nothing, and the agent was
    denied something it had earned.
    """

    kind = "latent_visibility"

    def __init__(self, unexpected: Iterable[str], missing: Iterable[str]) -> None:
        self.unexpected = sorted(unexpected)
        self.missing = sorted(missing)
        super().__init__(
            "latent task visibility does not match the event ledger: "
            f"released without an event {self.unexpected}, "
            f"withheld despite one {self.missing}"
        )

    def evidence(self) -> dict[str, Any]:
        return {"unexpected": self.unexpected, "missing": self.missing}


class UnknownEventError(IntegrityError):
    """The event ledger carries ids this scenario never declared."""

    kind = "unknown_event"

    def __init__(self, event_ids: Iterable[str]) -> None:
        self.event_ids = sorted(event_ids)
        super().__init__(f"event ledger carries undeclared events: {self.event_ids}")

    def evidence(self) -> dict[str, Any]:
        return {"event_ids": self.event_ids}


_REQUIRED_SECTIONS = (
    "tasks", "users", "projects", "milestones", "assignments",
    "scenario_events", "action_log", "reports",
)


def check_workspace_integrity(state: dict[str, Any]) -> None:
    """Raise if this episode cannot be read as evidence about the agent."""
    missing = [section for section in _REQUIRED_SECTIONS if section not in state]
    if missing:
        raise MissingSectionsError(missing)

    present = {task["task_id"] for task in state["tasks"]}
    vanished = SEEDED_TASK_IDS - present
    if vanished:
        raise DestroyedHistoryError(vanished)

    declared = set(WORKSPACE_MILESTONES)
    ledger = {event["event_id"] for event in state["scenario_events"]}
    if ledger - declared:
        raise UnknownEventError(ledger - declared)

    activated = {
        event["event_id"] for event in state["scenario_events"]
        if event.get("status") == "activated"
    }
    unexpected: list[str] = []
    withheld: list[str] = []
    for latent in state.get("latent_tasks", []):
        released = bool(latent.get("released"))
        earned = latent.get("event_id") in activated
        if released and not earned:
            unexpected.append(latent["task_id"])
        elif earned and not released:
            withheld.append(latent["task_id"])
    if unexpected or withheld:
        raise LatentVisibilityError(unexpected, withheld)


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------


def _events(state: dict[str, Any]) -> dict[str, str]:
    return {
        str(event["event_id"]): str(event.get("status", "pending"))
        for event in state.get("scenario_events", [])
    }


def _labels_of(task: dict[str, Any]) -> tuple[str, ...]:
    raw = task.get("labels")
    if isinstance(raw, list):
        return tuple(str(item) for item in raw)
    try:
        return tuple(json.loads(raw or "[]"))
    except (json.JSONDecodeError, TypeError):
        return ()


def side_effects(state: dict[str, Any]) -> list[str]:
    """Changes to the workspace nobody asked for, one label each.

    Counted per record and per field rather than per category, so the cost is
    proportional to what the episode actually did instead of being a flat fee
    for straying once.

    The five tasks the handover names are exempt on assignee and labels -- those
    are the fields it exists to change -- but not on status or project, which it
    never asks anyone to touch.
    """
    offenders: list[str] = []
    seeded = {row[0]: row for row in TRACKER_TASKS}
    for task in state.get("tasks", []):
        task_id = task["task_id"]
        row = seeded.get(task_id)
        if row is None:
            offenders.append(f"created:{task_id}")
            continue
        if task["status"] != row[5]:
            offenders.append(f"status_changed:{task_id}")
        if task["project_id"] != row[6]:
            offenders.append(f"moved:{task_id}")
        if task["milestone_id"] != row[7]:
            offenders.append(f"remilestoned:{task_id}")
        if task_id not in REASSIGNED_TASKS:
            # Reassigning someone the instruction never mentioned is the most
            # likely off-task mutation here, so leaving it out of the shaped
            # signal would hide the mistake the task most invites.
            if task["assignee_id"] != row[4]:
                offenders.append(f"reassigned:{task_id}")
            if _labels_of(task) != tuple(row[10]):
                offenders.append(f"relabelled:{task_id}")
    for violation in state.get("integrity_violations", []):
        offenders.append(f"integrity:{violation.get('kind', 'unknown')}")
    return offenders


def _evaluate_workspace(state: dict[str, Any]) -> dict[str, bool]:
    events = _events(state)
    return {event: events.get(event) == "activated" for event in WORKSPACE_MILESTONES}


def workspace_progress(state: dict[str, Any]) -> dict[str, Any]:
    """How far along the rollout is, and whether it is done.

    Progress is the fraction of observable milestones met, less the side-effect
    penalty. A rollout terminates when every milestone is met with no penalty.
    """
    met = _evaluate_workspace(state)
    offenders = side_effects(state)
    penalty = round(_SIDE_EFFECT_PER_ACTION * len(offenders), 6)
    progress = round(max(0.0, sum(met.values()) / len(met) - penalty), 6)
    return {
        "progress": progress,
        "milestones_met": met,
        "milestones_total": len(met),
        "side_effects": offenders,
        "penalty_total": penalty,
        "complete": all(met.values()) and not offenders,
    }


__all__ = [
    "DestroyedHistoryError", "IntegrityError", "LatentVisibilityError",
    "MissingSectionsError", "REASSIGNED_TASKS", "SEEDED_TASK_IDS",
    "UnknownEventError", "WORKSPACE_MILESTONES", "check_workspace_integrity",
    "side_effects", "workspace_progress",
]
