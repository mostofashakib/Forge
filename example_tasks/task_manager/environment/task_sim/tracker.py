"""The generic scenario evaluator: what the world does about what the agent did.

Every completed tool call passes through `track_action`, so this is also the one
place that observes the agent's whole action stream. It does two things and
nothing task-specific:

1. appends the call to the world's own `action_log`, and
2. evaluates every pending `ScenarioRule` against it, activating the events
   whose rules matched and releasing whatever those events carry.

Evaluation runs to a fixed point, so a closure resolves in the same call that
completed its last prerequisite rather than waiting for an unrelated action to
come along and nudge it.

Grading that reads a trajectory out of the agent's log directory is grading a
file the agent can write. The `action_log` lives in the world's database, which
the agent's container has no filesystem path to, so trajectory and side-effect
checks rest on evidence the graded party could not have authored.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any

from task_sim.clock import VIRTUAL_CLOCK

#: Trigger kinds the engine understands. A scenario that names anything else is
#: an authoring mistake, and is reported as one at seed time rather than
#: silently never firing.
TRIGGER_KINDS = frozenset(
    {"observed", "field_equals", "label_present", "tool_called", "all_of"}
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Predicate:
    """One clause of one rule, and what it saw."""

    name: str
    passed: bool
    candidates: tuple[str, ...] = ()
    matched: tuple[str, ...] = ()

    def render(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        if not self.candidates:
            return f"  {self.name}: {verdict}"
        return (
            f"  {self.name}: candidates={list(self.candidates)} "
            f"matched={list(self.matched)} {verdict}"
        )


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    event_id: str
    rule_id: str
    trigger: str
    actor: str
    action: str
    predicates: tuple[Predicate, ...]
    matched: bool
    event_before: str
    event_after: str
    released_tasks: tuple[str, ...] = ()
    pass_index: int = 0

    def render(self) -> str:
        lines = [
            "[event-eval]",
            f"event_id={self.event_id}",
            f"rule_id={self.rule_id}",
            f"trigger={self.trigger}",
            f"action={self.action} actor={self.actor}",
            "predicates:",
        ]
        lines.extend(predicate.render() for predicate in self.predicates)
        lines.append(f"final_match={'PASS' if self.matched else 'FAIL'}")
        if self.released_tasks:
            lines.append(f"released={list(self.released_tasks)}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Activation:
    event_id: str
    released_tasks: tuple[str, ...] = ()
    released_dependencies: tuple[str, ...] = ()


@dataclass
class EventTrace:
    """World-side only. Never attached to a tool result, never crosses a socket."""

    evaluations: list[RuleEvaluation] = field(default_factory=list)
    activations: list[Activation] = field(default_factory=list)


def _emit(trace: EventTrace) -> None:
    if os.environ.get("TASK_EVENT_TRACE", "") not in ("1", "true", "TRUE"):
        return
    for evaluation in trace.evaluations:
        print(evaluation.render(), file=sys.stderr)


# ---------------------------------------------------------------------------
# What one call looked like
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Action:
    tool_name: str
    actor_id: str
    task_id: str | None
    project_id: str | None
    observed: frozenset[str]
    ok: bool = True


def _observed_ids(result: Any) -> set[str]:
    """Every identifier in the serialized result the agent actually received.

    Walked over the response rather than over the rows a handler touched:
    search may inspect more internally, and a row that was not returned cannot
    have been observed.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("_id") or key == "id":
                    if isinstance(value, str) and value:
                        found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return found


def _describe_action(
    tool_name: str, payload: dict[str, Any], result: dict[str, Any], actor_id: str
) -> _Action:
    task_id = payload.get("task_id")
    project_id = payload.get("project_id")
    return _Action(
        tool_name=tool_name,
        actor_id=actor_id,
        task_id=task_id if isinstance(task_id, str) and task_id else None,
        project_id=project_id if isinstance(project_id, str) and project_id else None,
        observed=frozenset(_observed_ids(result)),
    )


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Rule:
    rule_id: str
    event_id: str
    trigger: str
    observed_ids: tuple[str, ...]
    table: str
    row_id: str
    field: str
    value: str
    labels: tuple[str, ...]
    tools: tuple[str, ...]
    requires_activated: tuple[str, ...]
    requires_pending: tuple[str, ...]


def _load_rules(connection: sqlite3.Connection) -> list[_Rule]:
    rows = connection.execute(
        "SELECT * FROM scenario_rules ORDER BY rule_id"
    ).fetchall()
    return [
        _Rule(
            rule_id=row["rule_id"],
            event_id=row["event_id"],
            trigger=row["trigger"],
            observed_ids=tuple(json.loads(row["observed_ids"])),
            table=row["table_name"] or "",
            row_id=row["row_id"] or "",
            field=row["field_name"] or "",
            value=row["field_value"] or "",
            labels=tuple(json.loads(row["labels"])),
            tools=tuple(json.loads(row["tools"])),
            requires_activated=tuple(json.loads(row["requires_activated"])),
            requires_pending=tuple(json.loads(row["requires_pending"])),
        )
        for row in rows
    ]


def _statuses(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        row["event_id"]: row["status"]
        for row in connection.execute("SELECT event_id, status FROM scenario_events")
    }


def _gates(rule: _Rule, statuses: dict[str, str]) -> list[Predicate]:
    """Ordering, expressed as what must already have happened and what must not.

    Defensive at runtime: a rule naming an event the ledger does not carry
    stays pending rather than raising, because a scenario-authoring mistake
    must never turn a valid tool call into an error the agent has to interpret.
    """
    predicates: list[Predicate] = []
    for event_id in rule.requires_activated:
        predicates.append(
            Predicate(f"requires_activated={event_id}", statuses.get(event_id) == "activated")
        )
    for event_id in rule.requires_pending:
        predicates.append(
            Predicate(f"requires_pending={event_id}", statuses.get(event_id) == "pending")
        )
    return predicates


def _task_row(connection: sqlite3.Connection, task_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
    ).fetchone()


def _evaluate(
    connection: sqlite3.Connection,
    rule: _Rule,
    statuses: dict[str, str],
    action: _Action,
) -> tuple[bool, tuple[Predicate, ...]]:
    predicates = _gates(rule, statuses)
    if not all(predicate.passed for predicate in predicates):
        return False, tuple(predicates)

    if rule.tools:
        hit = action.tool_name in rule.tools
        predicates.append(
            Predicate("tool", hit, rule.tools, (action.tool_name,) if hit else ())
        )
        if not hit:
            return False, tuple(predicates)

    if rule.trigger == "all_of":
        # The gates were the whole rule. Reaching here means they all held.
        return True, tuple(predicates)

    if rule.trigger == "tool_called":
        # `tools` above was the whole rule; a rule with neither is malformed
        # and never fires rather than firing on everything.
        matched = bool(rule.tools)
        predicates.append(Predicate("tool_called", matched))
        return matched, tuple(predicates)

    if rule.trigger == "observed":
        missing = [item for item in rule.observed_ids if item not in action.observed]
        matched = bool(rule.observed_ids) and not missing
        predicates.append(
            Predicate(
                "observed",
                matched,
                rule.observed_ids,
                tuple(item for item in rule.observed_ids if item in action.observed),
            )
        )
        return matched, tuple(predicates)

    if rule.trigger == "field_equals":
        if rule.table != "tasks":
            predicates.append(Predicate(f"unsupported_table={rule.table}", False))
            return False, tuple(predicates)
        row = _task_row(connection, rule.row_id)
        actual = "" if row is None else ("" if row[rule.field] is None else str(row[rule.field]))
        matched = actual == rule.value
        predicates.append(
            Predicate(f"{rule.row_id}.{rule.field}", matched, (rule.value,), (actual,))
        )
        return matched, tuple(predicates)

    if rule.trigger == "label_present":
        row = _task_row(connection, rule.row_id)
        present = tuple(json.loads(row["labels"] or "[]")) if row is not None else ()
        matched = bool(rule.labels) and all(label in present for label in rule.labels)
        predicates.append(Predicate(f"{rule.row_id}.labels", matched, rule.labels, present))
        return matched, tuple(predicates)

    predicates.append(Predicate(f"unknown_trigger={rule.trigger}", False))
    return False, tuple(predicates)


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def _publishes(connection: sqlite3.Connection, event_id: str) -> bool:
    """Whether this event puts anything into the world when it fires."""
    for table in ("latent_tasks", "latent_dependencies"):
        row = connection.execute(
            f"SELECT 1 FROM {table} WHERE event_id = ? AND released = 0 LIMIT 1",
            (event_id,),
        ).fetchone()
        if row is not None:
            return True
    return False


def _activate(connection: sqlite3.Connection, event_id: str) -> Activation:
    """Mark one event activated and publish whatever it carries.

    Only an event that actually does something takes an instant. A scheduled
    event lands on the seed calendar, and one that publishes rows is a thing
    that happened in the world and gets a time of its own -- but an
    observation-only event records that the agent noticed something, and
    noticing is not an act of the world. Advancing for those let a read move the
    clock, which is precisely what the clock is built not to do.
    """
    row = connection.execute(
        "SELECT scheduled_step FROM scenario_events WHERE event_id = ?", (event_id,)
    ).fetchone()
    scheduled = None if row is None else row["scheduled_step"]
    if scheduled is not None:
        activated_ms = VIRTUAL_CLOCK.advance_to(connection, int(scheduled))
    elif _publishes(connection, event_id):
        activated_ms = VIRTUAL_CLOCK.advance(connection)
    else:
        activated_ms = VIRTUAL_CLOCK.now(connection)

    connection.execute(
        "UPDATE scenario_events SET status = 'activated', activated_ms = ? WHERE event_id = ?",
        (activated_ms, event_id),
    )

    released_tasks: list[str] = []
    for latent in connection.execute(
        "SELECT * FROM latent_tasks WHERE event_id = ? AND released = 0 ORDER BY task_id",
        (event_id,),
    ).fetchall():
        connection.execute(
            "INSERT INTO tasks ("
            "  task_id, title, description, creator_id, assignee_id, status,"
            "  created_at_ms, updated_at_ms, project_id, milestone_id, due_at_ms,"
            "  priority, labels, deleted"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 0)",
            (
                latent["task_id"], latent["title"], latent["description"],
                latent["creator_id"], latent["assignee_id"], latent["status"],
                activated_ms, activated_ms, latent["project_id"],
                latent["milestone_id"], latent["priority"], latent["labels"],
            ),
        )
        connection.execute(
            "UPDATE latent_tasks SET released = 1, released_ms = ? WHERE task_id = ?",
            (activated_ms, latent["task_id"]),
        )
        released_tasks.append(latent["task_id"])

    released_dependencies: list[str] = []
    for latent in connection.execute(
        "SELECT * FROM latent_dependencies WHERE event_id = ? AND released = 0 ORDER BY dep_id",
        (event_id,),
    ).fetchall():
        connection.execute(
            "INSERT INTO task_dependencies VALUES (?, ?, ?)",
            (latent["dep_id"], latent["task_id"], latent["depends_on_task_id"]),
        )
        connection.execute(
            "UPDATE latent_dependencies SET released = 1 WHERE dep_id = ?", (latent["dep_id"],)
        )
        released_dependencies.append(latent["dep_id"])

    return Activation(event_id, tuple(released_tasks), tuple(released_dependencies))


def _record(connection: sqlite3.Connection, action: _Action) -> None:
    connection.execute(
        """
        INSERT INTO action_log (ms, actor_id, tool, task_id, project_id, observed_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            VIRTUAL_CLOCK.now(connection),
            action.actor_id,
            action.tool_name,
            action.task_id,
            action.project_id,
            len(action.observed),
        ),
    )


def track_action(
    connection: sqlite3.Connection,
    tool_name: str,
    payload: dict[str, Any],
    result: dict[str, Any],
    actor_id: str,
    trace: EventTrace | None = None,
) -> list[str]:
    """Record one completed operation and apply everything it causes.

    The return value is verifier and debug metadata; it is deliberately not
    added to the agent-facing tool response, and neither is `trace`.
    """
    action = _describe_action(tool_name, payload, result, actor_id)
    _record(connection, action)
    collector = trace if trace is not None else EventTrace()

    activated: list[str] = []
    progressing = True
    pass_index = 0
    while progressing:
        progressing = False
        statuses = _statuses(connection)
        for rule in _load_rules(connection):
            if statuses.get(rule.event_id) != "pending":
                continue
            matched, predicates = _evaluate(connection, rule, statuses, action)
            activation = _activate(connection, rule.event_id) if matched else None
            collector.evaluations.append(
                RuleEvaluation(
                    event_id=rule.event_id,
                    rule_id=rule.rule_id,
                    trigger=rule.trigger,
                    actor=action.actor_id,
                    action=action.tool_name,
                    predicates=predicates,
                    matched=matched,
                    event_before="pending",
                    event_after="activated" if activation else "pending",
                    released_tasks=activation.released_tasks if activation else (),
                    pass_index=pass_index,
                )
            )
            if activation is not None:
                collector.activations.append(activation)
                activated.append(rule.event_id)
                progressing = True
                break
        pass_index += 1

    if trace is None:
        _emit(collector)
    return activated


__all__ = [
    "Activation", "EventTrace", "Predicate", "RuleEvaluation", "TRIGGER_KINDS",
    "track_action",
]
