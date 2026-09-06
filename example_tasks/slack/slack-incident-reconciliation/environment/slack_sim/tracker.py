"""The tracker: it watches every agent action and lets the workspace react.

Every completed tool call passes through here. The tracker reads what the action
was, weighs it against the scenario's rules, and publishes whatever the world
says in response -- which makes it both the reactive engine and the one place
that sees the agent's whole action stream.

Rules stored in the database say what the agent has to do, and events say what
other people then say. Nothing here knows which task is loaded -- the rules come
out of `scenario_rules`, the replies out of `latent_messages`, and both are
seeded from a `Scenario`.

Two properties are worth stating because scoring depends on them.

*Release follows observation.* The engine runs after a tool call has produced
its result, so the call that satisfies a rule never sees the messages it
released; they are ordinary Slack messages from the next call onward.

*Activation is once.* An event moves from `pending` to `activated` under a
guarded UPDATE, so a repeated action cannot publish the same replies twice.

Diagnostics
-----------
Every rule considered for an action records why it matched or did not, one
predicate at a time, into an optional `EventTrace`. That is what makes a failed
episode attributable: "the reply named 9:30 but no clarification keyword" is a
different finding from "the reply never reached the thread", and both are
different from "the event fired but nothing became visible".

The trace is world-side only. It is never attached to a tool result, never
crosses `agent.sock`, and is off unless a caller asks for it or the environment
sets ``SLACK_EVENT_TRACE``.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any

from slack_sim.clock import VIRTUAL_CLOCK
from slack_sim.identity import LOGGED_IN_USER
from slack_sim.models import ScenarioRule

#: Set to any non-empty value to have traces written to stderr as they happen.
#: The agent's container cannot see this; the world writes it to its own log.
TRACE_ENV_VAR = "SLACK_EVENT_TRACE"


def _normalize(body: str) -> str:
    return re.sub(r"[\s.!,]+$", "", body.strip().lower())


def _matched_words(body: str, group: tuple[str, ...]) -> list[str]:
    """Which candidates in one group the body actually contains."""
    text = body.lower()
    return [word for word in group if word in text]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Predicate:
    """One condition a rule imposes, and whether this action satisfied it."""

    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def render(self, indent: str = "  ") -> str:
        verdict = "PASS" if self.passed else "FAIL"
        if not self.detail:
            return f"{indent}{self.name}: {verdict}"
        lines = [f"{indent}{self.name}:"]
        for key, value in self.detail.items():
            lines.append(f"{indent}  {key}={json.dumps(value, default=str)}")
        lines.append(f"{indent}  result={verdict}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """A single rule weighed against a single action."""

    event_id: str
    rule_id: str
    trigger: str
    actor: str
    action: str
    conversation: str | None
    thread: str | None
    raw_text: str | None
    predicates: tuple[Predicate, ...]
    matched: bool
    event_before: str
    event_after: str
    activated_messages: tuple[str, ...] = ()
    pass_index: int = 0

    @property
    def failed_predicates(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.predicates if not p.passed)

    def render(self) -> str:
        lines = [
            "[event-eval]",
            f"event_id={self.event_id}",
            f"rule_id={self.rule_id}",
            f"trigger={self.trigger}",
            f"actor={self.actor}",
            f"action={self.action}",
            f"conversation={self.conversation}",
            f"thread={self.thread}",
            f"raw_text={json.dumps(self.raw_text)}",
            f"normalized_text={json.dumps(_normalize(self.raw_text) if self.raw_text else None)}",
            "",
            "predicates:",
            *(predicate.render() for predicate in self.predicates),
            "",
            f"final_match={'PASS' if self.matched else 'FAIL'}",
            f"event_before={self.event_before}",
            f"event_after={self.event_after}",
            f"activated_messages={list(self.activated_messages)}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Activation:
    """What one event actually published when it fired."""

    event_id: str
    status_before: str
    status_after: str
    cause_action_id: str | None
    latent_records_activated: tuple[str, ...]
    notifications_activated: tuple[str, ...]
    membership_mutations: tuple[str, ...]
    activated_ts: str

    def render(self) -> str:
        return "\n".join(
            [
                "[event-activation]",
                f"event_id={self.event_id}",
                f"transition={self.status_before}->{self.status_after}",
                f"cause_action_id={self.cause_action_id}",
                f"latent_records_activated={list(self.latent_records_activated)}",
                f"notifications_activated={list(self.notifications_activated)}",
                f"membership_mutations={list(self.membership_mutations)}",
                f"activated_ts={self.activated_ts}",
            ]
        )


@dataclass
class EventTrace:
    """Everything one tool call caused the engine to consider."""

    evaluations: list[RuleEvaluation] = field(default_factory=list)
    activations: list[Activation] = field(default_factory=list)

    def for_rule(self, rule_id: str) -> list[RuleEvaluation]:
        return [item for item in self.evaluations if item.rule_id == rule_id]

    def for_event(self, event_id: str) -> list[RuleEvaluation]:
        return [item for item in self.evaluations if item.event_id == event_id]

    def considered(self, rule_id: str) -> bool:
        """Did the evaluator weigh this rule at all for this action?

        A rule whose event has already fired is never reloaded, so "not
        considered" and "considered and rejected" are genuinely different
        findings about a trajectory.
        """
        return bool(self.for_rule(rule_id))

    def matched(self, rule_id: str) -> bool:
        return any(item.matched for item in self.for_rule(rule_id))

    def activated_events(self) -> list[str]:
        return [item.event_id for item in self.activations]

    def render(self) -> str:
        blocks = [item.render() for item in self.evaluations]
        blocks.extend(item.render() for item in self.activations)
        return "\n\n".join(blocks)


def _emit(trace: EventTrace) -> None:
    if os.environ.get(TRACE_ENV_VAR) and (trace.evaluations or trace.activations):
        print(trace.render(), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Rule loading and evaluation
# ---------------------------------------------------------------------------


def _load_rules(connection: sqlite3.Connection) -> list[ScenarioRule]:
    """Rules for events that could still fire, in a deterministic order."""
    rows = connection.execute(
        """
        SELECT r.rule_id, r.event_id, r.trigger, r.thread_id, r.channel_id, r.exact_body,
               r.keyword_groups, r.observed_ids, r.requires_activated, r.requires_pending,
               r.recipient_ids, r.tools
        FROM scenario_rules r
        JOIN scenario_events e ON e.event_id = r.event_id
        WHERE e.status = 'pending'
        ORDER BY r.rule_id
        """
    ).fetchall()
    return [
        ScenarioRule(
            rule_id=row[0], event_id=row[1], trigger=row[2], thread_id=row[3],
            channel_id=row[4], exact_body=row[5],
            keyword_groups=tuple(tuple(group) for group in json.loads(row[6])),
            observed_ids=tuple(json.loads(row[7])),
            requires_activated=tuple(json.loads(row[8])),
            requires_pending=tuple(json.loads(row[9])),
            recipient_ids=tuple(json.loads(row[10])),
            tools=tuple(json.loads(row[11])),
        )
        for row in rows
    ]


def _statuses(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row[0]): str(row[1])
        for row in connection.execute("SELECT event_id, status FROM scenario_events")
    }


@dataclass(frozen=True, slots=True)
class _Action:
    """One completed tool call, in the terms the rules are written in."""

    tool_name: str
    actor_id: str
    conversation: str | None
    reply_thread: str | None
    #: The thread this call addressed, whatever the tool. Recorded in the
    #: action log so a reader of six different threads is distinguishable
    #: from six reads of one; never consulted when matching rules, where a
    #: read must not be able to satisfy a reply trigger.
    addressed_thread: str | None
    reply_body: str | None
    #: Whatever text the action carried, even when no rule can read it.
    #: Diagnostics rely on it to say "the wording was there, the addressing was
    #: not", instead of blaming the wording for both.
    action_body: str | None
    #: Who a direct message went to, when the action was one.
    dm_recipient: str | None
    observed: frozenset[str]
    joined_channel: str | None
    trigger_message_id: str | None

    @property
    def is_agent(self) -> bool:
        return self.actor_id == LOGGED_IN_USER.user_id


def _activate(
    connection: sqlite3.Connection, event_id: str, trigger_message_id: str | None
) -> Activation | None:
    """Publish one event's messages and mark it activated, exactly once."""
    row = connection.execute(
        "SELECT status, scheduled_step FROM scenario_events WHERE event_id = ?", (event_id,)
    ).fetchone()
    if row is None or row[0] != "pending":
        return None
    scheduled_step = row[1]
    if scheduled_step is None:
        now_ts = VIRTUAL_CLOCK.advance(connection)
    else:
        now_ts = VIRTUAL_CLOCK.advance_to(connection, int(scheduled_step))

    messages = connection.execute(
        """
        SELECT message_id, channel_id, author_id, body, thread_parent_id,
               reply_to_message_id
        FROM latent_messages WHERE event_id = ? ORDER BY ordinal, message_id
        """,
        (event_id,),
    ).fetchall()
    # Resolve every parent before writing anything. An event whose replies
    # answer a message that is not visible yet is not ready to fire, and the
    # agent must not see a valid tool call fail because a scenario left an
    # ordering implicit: the event simply stays pending. Authoring errors are
    # caught at test time by the ordering invariant, not at the agent's expense.
    #
    # An event routinely publishes a root and its replies together, so a parent
    # this same event is about to insert counts as resolvable; ordinal order
    # puts it in the table first.
    own = {str(message[0]) for message in messages}
    parents: dict[str, str] = {}
    for message in messages:
        parent_id = message[4]
        if not parent_id or parent_id in parents or parent_id in own:
            continue
        parent = connection.execute(
            "SELECT ts FROM messages WHERE message_id = ?", (parent_id,)
        ).fetchone()
        if parent is None:
            return None
        parents[parent_id] = parent[0]

    published: list[str] = []
    for message in messages:
        parent_id = message[4]
        thread_ts = None
        if parent_id:
            if parent_id not in parents:
                row = connection.execute(
                    "SELECT ts FROM messages WHERE message_id = ?", (parent_id,)
                ).fetchone()
                if row is None:
                    return None
                parents[parent_id] = row[0]
            thread_ts = parents[parent_id]
        connection.execute(
            """
            INSERT INTO messages (
                message_id, channel_id, author_id, body, ts,
                thread_parent_id, thread_ts, edited_ts, reply_to_message_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (message[0], message[1], message[2], message[3], now_ts,
             parent_id, thread_ts, message[5]),
        )
        published.append(str(message[0]))
    notified: list[str] = []
    for row in connection.execute(
        "SELECT notification_id, user_id, kind, message_id, conversation_id "
        "FROM latent_notifications WHERE event_id = ? ORDER BY notification_id",
        (event_id,),
    ).fetchall():
        connection.execute(
            "INSERT INTO notifications (notification_id, user_id, kind, message_id, "
            "conversation_id, created_ts, read_ts) VALUES (?, ?, ?, ?, ?, ?, NULL)",
            (row[0], row[1], row[2], row[3], row[4], now_ts),
        )
        notified.append(str(row[0]))
    joined: list[str] = []
    for row in connection.execute(
        "SELECT membership_id, channel_id, user_id, role FROM latent_memberships "
        "WHERE event_id = ? ORDER BY membership_id",
        (event_id,),
    ).fetchall():
        connection.execute(
            "INSERT OR IGNORE INTO memberships VALUES (?, ?, ?, ?, ?)",
            (row[0], row[1], row[2], row[3], now_ts),
        )
        joined.append(str(row[0]))
    connection.execute(
        """
        UPDATE scenario_events
        SET status = 'activated', activated_ts = ?, trigger_message_id = ?
        WHERE event_id = ? AND status = 'pending'
        """,
        (now_ts, trigger_message_id, event_id),
    )
    return Activation(
        event_id=event_id,
        status_before="pending",
        status_after="activated",
        cause_action_id=trigger_message_id,
        latent_records_activated=tuple(published),
        notifications_activated=tuple(notified),
        membership_mutations=tuple(joined),
        activated_ts=str(now_ts),
    )


def _observed_message_ids(result: Any) -> set[str]:
    """Ids the agent actually received.

    Only the serialized result counts. A search may read more rows than it
    returns, and a row the agent never saw must not satisfy a rule that exists
    to prove the agent went and looked.
    """
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            message_id = value.get("id")
            if isinstance(message_id, str):
                found.add(message_id)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(result)
    return found


def _prerequisite_predicates(
    rule: ScenarioRule, statuses: dict[str, str]
) -> list[Predicate]:
    predicates: list[Predicate] = []
    if rule.requires_activated:
        unmet = [e for e in rule.requires_activated if statuses.get(e) != "activated"]
        predicates.append(
            Predicate(
                "requires_activated",
                not unmet,
                {
                    "required": list(rule.requires_activated),
                    "not_yet_activated": unmet,
                    "statuses": {e: statuses.get(e, "unknown") for e in rule.requires_activated},
                },
            )
        )
    if rule.requires_pending:
        broken = [e for e in rule.requires_pending if statuses.get(e) != "pending"]
        predicates.append(
            Predicate(
                "requires_pending",
                not broken,
                {"required": list(rule.requires_pending), "no_longer_pending": broken},
            )
        )
    return predicates


def _trigger_predicates(rule: ScenarioRule, action: _Action) -> list[Predicate]:
    """The conditions specific to this trigger kind, all of them evaluated.

    Nothing short-circuits: a caller investigating a failure needs to know that
    the thread was right and only the wording was wrong, which a first-failure
    exit would hide.
    """
    if rule.trigger == "all_of":
        return []

    if rule.trigger == "observed":
        seen = sorted(action.observed & set(rule.observed_ids))
        return [
            Predicate(
                "observed_required_evidence",
                bool(seen),
                {"candidates": list(rule.observed_ids), "matched": seen},
            )
        ]

    if rule.trigger == "channel_joined":
        return [
            Predicate("action_type", action.tool_name == "join_channel",
                      {"expected": "join_channel", "actual": action.tool_name}),
            Predicate("actor_is_the_agent", action.is_agent,
                      {"expected": LOGGED_IN_USER.user_id, "actual": action.actor_id}),
            Predicate("channel", action.joined_channel == rule.channel_id,
                      {"expected": rule.channel_id, "actual": action.joined_channel}),
        ]

    # `reply_exact` stays anchored to its thread: the task instructs a verbatim
    # string, so there is nothing to be generous about.
    if rule.trigger == "reply_exact":
        body = action.reply_body or ""
        return [
            Predicate("action_type", action.tool_name == "reply_to_thread",
                      {"expected": "reply_to_thread", "actual": action.tool_name}),
            Predicate("actor_is_the_agent", action.is_agent,
                      {"expected": LOGGED_IN_USER.user_id, "actual": action.actor_id}),
            Predicate(
                f"thread_parent={rule.thread_id}",
                action.reply_thread == rule.thread_id,
                {
                    "expected_thread": rule.thread_id,
                    "actual_thread": action.reply_thread,
                    "actual_conversation": action.conversation,
                },
            ),
            Predicate(
                "body_matches_exactly",
                action.reply_body is not None
                and _normalize(body) == _normalize(rule.exact_body or ""),
                {
                    "expected": _normalize(rule.exact_body or ""),
                    "actual": _normalize(body) if action.reply_body is not None else None,
                },
            ),
        ]

    # An ask can be raised in more than one defensible place. The rule declares
    # which ones it listens to, and reaching any of them is enough; what the
    # agent has to demonstrate is still decided by the keyword groups below.
    tools = rule.accepted_tools()
    carried = action.tool_name in tools and action.is_agent
    matched_addresses = []
    if rule.thread_id and action.reply_thread == rule.thread_id:
        matched_addresses.append(f"thread:{rule.thread_id}")
    if rule.channel_id and action.conversation == rule.channel_id:
        matched_addresses.append(f"channel:{rule.channel_id}")
    if rule.recipient_ids and action.dm_recipient in rule.recipient_ids:
        matched_addresses.append(f"dm:{action.dm_recipient}")

    predicates = [
        Predicate("action_type", carried,
                  {"expected": list(tools), "actual": action.tool_name}),
        Predicate("actor_is_the_agent", action.is_agent,
                  {"expected": LOGGED_IN_USER.user_id, "actual": action.actor_id}),
        Predicate(
            "address",
            bool(matched_addresses) and carried,
            {
                "accepted": [
                    *( [f"thread:{rule.thread_id}"] if rule.thread_id else [] ),
                    *( [f"channel:{rule.channel_id}"] if rule.channel_id else [] ),
                    *( [f"dm:{person}" for person in rule.recipient_ids] ),
                ],
                "matched": matched_addresses,
                "actual_thread": action.reply_thread,
                "actual_conversation": action.conversation,
                "actual_recipient": action.dm_recipient,
            },
        ),
    ]

    # The body of whichever operation carried the ask, so a DM and a thread
    # reply are read the same way once the rule agrees to listen to both.
    body = action.action_body if carried else None
    for index, group in enumerate(rule.keyword_groups, start=1):
        matched = _matched_words(body or "", group) if body is not None else []
        detail: dict[str, Any] = {"candidates": list(group), "matched": matched}
        if body is None:
            detail["note"] = (
                f"not evaluated: this rule listens to {list(tools)}, not "
                f"{action.tool_name}"
            )
            detail["matched_in_action_body"] = _matched_words(
                action.action_body or "", group
            )
        predicates.append(Predicate(f"keyword_group_{index}", bool(matched), detail))
    if not rule.keyword_groups:
        predicates.append(Predicate("keyword_groups_defined", False, {"groups": []}))
    return predicates


def _evaluate(
    rule: ScenarioRule, statuses: dict[str, str], action: _Action
) -> tuple[bool, list[Predicate], str | None]:
    """Weigh one rule and report every predicate, not just the first failure.

    The verdict is identical to evaluating the same conditions with early
    exits; only the reporting differs.
    """
    predicates = _prerequisite_predicates(rule, statuses) + _trigger_predicates(rule, action)
    matched = all(predicate.passed for predicate in predicates)

    evidence_id: str | None = None
    if rule.trigger == "observed":
        seen = sorted(action.observed & set(rule.observed_ids))
        evidence_id = seen[0] if seen else None
    return matched, predicates, evidence_id


def _describe_action(
    tool_name: str, payload: dict[str, Any], result: dict[str, Any], actor_id: str
) -> _Action:
    """Reduce one completed tool call to what the rules can see of it."""
    # Diagnostic only: where the action landed, so a trace can say "this was a
    # DM to Nina" rather than leaving the reader to infer it. Matching never
    # reads this field.
    conversation = None
    posted = result.get("message") if isinstance(result, dict) else None
    sources: tuple[dict[str, Any], ...] = tuple(
        source for source in (posted if isinstance(posted, dict) else None, result, payload)
        if isinstance(source, dict)
    )
    for source in sources:
        for key in ("channel_id", "chat_id", "conversation_id"):
            value = source.get(key)
            if isinstance(value, str) and value:
                conversation = value
                break
        if conversation:
            break

    addressed_thread = None
    if tool_name == "get_thread_replies":
        thread_ts = payload.get("thread_ts") or payload.get("thread_parent_id")
        addressed_thread = str(thread_ts) if thread_ts else None

    reply_thread = reply_body = trigger_id = joined_channel = None
    if tool_name == "join_channel" and actor_id == LOGGED_IN_USER.user_id:
        joined_channel = str(result.get("channel_id") or "") or None
    if tool_name == "reply_to_thread" and actor_id == LOGGED_IN_USER.user_id:
        reply_thread = str(payload.get("thread_parent_id", ""))
        reply_body = str(payload.get("body", ""))
        trigger_id = str(result.get("id") or "") or None

    raw_body = payload.get("body")
    recipient = payload.get("recipient_id")
    return _Action(
        tool_name=tool_name,
        actor_id=actor_id,
        conversation=conversation,
        reply_thread=reply_thread,
        addressed_thread=addressed_thread or reply_thread,
        reply_body=reply_body,
        action_body=raw_body if isinstance(raw_body, str) else None,
        dm_recipient=recipient if isinstance(recipient, str) and recipient else None,
        observed=frozenset(_observed_message_ids(result)),
        joined_channel=joined_channel,
        trigger_message_id=trigger_id,
    )


def _record(connection: sqlite3.Connection, action: _Action) -> None:
    """Append what the agent just did to the world's own log.

    Grading that reads a trajectory out of the agent's log directory is grading
    a file the agent can write. This log lives in the environment's database,
    which the agent's container has no path to, so trajectory and side-effect
    checks rest on evidence the graded party could not have authored.
    """
    connection.execute(
        """
        INSERT INTO action_log (
            ts, actor_id, tool, conversation_id, thread_id, message_id, body_length
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            VIRTUAL_CLOCK.now(connection),
            action.actor_id,
            action.tool_name,
            action.conversation,
            action.addressed_thread,
            action.trigger_message_id,
            len(action.action_body or ""),
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
    """Record one completed Slack operation and apply everything it causes.

    Evaluation runs to a fixed point so that a rule depending only on other
    events resolves in the same call that completed its last prerequisite,
    rather than waiting for an unrelated action to come along and nudge it.

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
            matched, predicates, evidence_id = _evaluate(rule, statuses, action)
            activation: Activation | None = None
            if matched:
                activation = _activate(
                    connection, rule.event_id, evidence_id or action.trigger_message_id
                )
            collector.evaluations.append(
                RuleEvaluation(
                    event_id=rule.event_id,
                    rule_id=rule.rule_id,
                    trigger=rule.trigger,
                    actor=action.actor_id,
                    action=action.tool_name,
                    conversation=action.conversation,
                    thread=action.reply_thread,
                    raw_text=action.reply_body,
                    predicates=tuple(predicates),
                    matched=matched,
                    event_before="pending",
                    event_after="activated" if activation else "pending",
                    activated_messages=activation.latent_records_activated if activation else (),
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
