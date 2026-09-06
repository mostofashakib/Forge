#!/usr/bin/env python3
"""Reusable instrumentation for investigating latent-event activation.

An episode that ends badly has several possible causes, and from the outside
they look identical -- the workspace simply never changed. This module exists to
tell them apart:

  1. the agent never took an action the rule could even consider;
  2. it took a reasonable action that the rule rejected;
  3. the rule matched and the event fired, but the released state never became
     observable through the tools the agent actually uses;
  4. something else in propagation is broken.

`Workspace` drives the world exactly the way the socket server does -- through
`execute_tool`, so handlers, the transition engine, notifications and the
virtual clock all run -- while capturing the evaluator's reasoning. `probe`
then asks every agent-facing surface whether a message is visible, so a claim
like "LAT021 is out" is checked against history, threads, search, notifications
and thread metadata rather than against the events table alone.

Nothing here is shipped to the agent's image or exposed through a Slack tool.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slack_sim.identity import LOGGED_IN_USER
from slack_sim.service import execute_tool, export_state, seed_database
from slack_sim.tracker import EventTrace, RuleEvaluation

BEN = LOGGED_IN_USER.user_id


@dataclass(frozen=True, slots=True)
class Action:
    """One thing an actor did, in the terms a trajectory records it."""

    tool: str
    payload: dict[str, Any]
    actor: str = BEN
    label: str = ""

    def describe(self) -> str:
        thread = self.payload.get("thread_parent_id")
        where = thread or self.payload.get("channel_id") or self.payload.get("recipient_id") or "-"
        body = self.payload.get("body")
        rendered = f"{self.tool} [{where}]"
        return f"{rendered} {body!r}" if body else rendered


@dataclass
class Outcome:
    """What one action did to the world, and what the evaluator thought."""

    action: Action
    result: dict[str, Any]
    trace: EventTrace
    error: Exception | None = None

    @property
    def activated(self) -> list[str]:
        return self.trace.activated_events()

    def evaluation(self, rule_id: str) -> RuleEvaluation | None:
        """The last time this rule was weighed for this action, if it was."""
        weighed = self.trace.for_rule(rule_id)
        return weighed[-1] if weighed else None


@dataclass
class Visibility:
    """Whether one message can be reached through each agent-facing surface."""

    message_id: str
    history_visible: bool
    thread_visible: bool
    search_visible: bool
    notification_visible: bool
    reply_count_includes: bool
    latest_reply_is: bool
    body: str | None = None
    author: str | None = None
    surfaces: dict[str, bool] = field(default_factory=dict)

    @property
    def anywhere(self) -> bool:
        return any(self.surfaces.values())

    @property
    def everywhere(self) -> bool:
        return all(self.surfaces.values())

    def render(self) -> str:
        lines = ["[visibility-check]", f"message={self.message_id}"]
        lines += [f"{name}={str(value).lower()}" for name, value in self.surfaces.items()]
        return "\n".join(lines)


class Workspace:
    """A freshly seeded episode, driven the way the socket server drives it."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or tempfile.mkdtemp(prefix="event-diagnostics-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "slack.db"
        seed_database(self.db, self.root / "seed.sql")

    # -- driving ---------------------------------------------------------

    def act(self, action: Action) -> Outcome:
        """Run one action, capturing the evaluator's reasoning.

        A tool that refuses is an outcome, not a crash: "the agent tried to
        reply in a channel it had not joined" is exactly the kind of finding
        this module exists to record.
        """
        trace = EventTrace()
        try:
            result = execute_tool(self.db, action.tool, dict(action.payload), action.actor, trace)
        except Exception as error:  # noqa: BLE001 - the refusal is the datum
            return Outcome(action=action, result={}, trace=trace, error=error)
        return Outcome(action=action, result=result, trace=trace)

    def reply(self, thread: str, body: str, actor: str = BEN) -> Outcome:
        return self.act(Action("reply_to_thread",
                               {"thread_parent_id": thread, "body": body}, actor))

    def post(self, channel: str, body: str, actor: str = BEN) -> Outcome:
        return self.act(Action("post_message", {"channel_id": channel, "body": body}, actor))

    def dm(self, recipient: str, body: str, actor: str = BEN) -> Outcome:
        return self.act(Action("send_dm_message", {"recipient_id": recipient, "body": body}, actor))

    def read_channel(self, channel: str, actor: str = BEN) -> Outcome:
        return self.act(Action("get_channel_messages", {"channel_id": channel}, actor))

    def search(self, query: str, actor: str = BEN) -> Outcome:
        return self.act(Action("search_messages", {"query": query}, actor))

    def join(self, channel: str, actor: str = BEN) -> Outcome:
        return self.act(Action("join_channel", {"channel_id": channel}, actor))

    # -- inspecting ------------------------------------------------------

    def event_status(self, event_id: str) -> str:
        for event in export_state(self.db)["scenario_events"]:
            if event["event_id"] == event_id:
                return str(event["status"])
        raise KeyError(f"no such scenario event: {event_id}")

    def event_statuses(self) -> dict[str, str]:
        return {e["event_id"]: e["status"] for e in export_state(self.db)["scenario_events"]}

    def stored_messages(self, message_id: str) -> list[dict[str, Any]]:
        """Rows in the database, whether or not a tool would surface them."""
        return [m for m in export_state(self.db)["messages"] if m["message_id"] == message_id]

    def thread_ts(self, channel: str, message_id: str, actor: str = BEN) -> str | None:
        page = execute_tool(self.db, "get_channel_messages", {"channel_id": channel}, actor)
        for message in page["messages"]:
            if message["id"] == message_id:
                return str(message.get("thread_ts") or message["ts"])
        return None

    def probe(
        self,
        message_id: str,
        channel: str,
        thread_parent: str | None = None,
        search_terms: str = "",
        notification_id: str | None = None,
        actor: str = BEN,
    ) -> Visibility:
        """Ask every agent-facing surface whether `message_id` is reachable.

        Reads go through the tools rather than SQL on purpose. A message that
        exists in the table but never appears in `get_thread_replies` is
        invisible to the agent, and it is the agent's view that decides whether
        an episode was winnable.
        """
        in_history = False
        body = author = None
        cursor: str | None = None
        while True:
            payload: dict[str, Any] = {"channel_id": channel}
            if cursor:
                payload["cursor"] = cursor
            try:
                page = execute_tool(self.db, "get_channel_messages", payload, actor)
            except Exception:  # noqa: BLE001 - not a member is "not visible"
                page = {"messages": [], "next_cursor": None}
            for message in page["messages"]:
                if message["id"] == message_id:
                    in_history, body = True, message["text"]
                    author = message["user_id"]
            cursor = page.get("next_cursor")
            if not cursor:
                break

        in_thread = False
        reply_count_includes = latest_reply_is = False
        if thread_parent:
            root_ts = self.thread_ts(channel, thread_parent, actor)
            if root_ts:
                try:
                    replies = execute_tool(
                        self.db,
                        "get_thread_replies",
                        {"channel_id": channel, "thread_ts": root_ts},
                        actor,
                    )
                except Exception:  # noqa: BLE001
                    replies = {"messages": []}
                for message in replies.get("messages", []):
                    if message["id"] == message_id:
                        in_thread, body = True, message["text"]
                        author = message["user_id"]
                parent = next(
                    (m for m in self._channel_page(channel, actor) if m["id"] == thread_parent),
                    None,
                )
                if parent is not None:
                    ids = [m["id"] for m in replies.get("messages", []) if m["id"] != thread_parent]
                    reply_count_includes = int(parent.get("reply_count") or 0) >= len(
                        [i for i in ids if i == message_id]
                    ) and message_id in ids
                    latest = parent.get("latest_reply")
                    latest_reply_is = bool(latest) and any(
                        m["id"] == message_id and str(m["ts"]) == str(latest)
                        for m in replies.get("messages", [])
                    )

        in_search = False
        if search_terms:
            try:
                found = execute_tool(self.db, "search_messages", {"query": search_terms}, actor)
                in_search = any(m["id"] == message_id for m in found.get("matches", []))
            except Exception:  # noqa: BLE001
                in_search = False

        notified = False
        listing = execute_tool(self.db, "list_notifications", {}, actor)
        for notification in listing.get("notifications", []):
            if notification_id and notification["notification_id"] == notification_id:
                notified = True
            elif not notification_id and notification.get("message_id") == message_id:
                notified = True

        surfaces = {
            "history_visible": in_history,
            "thread_visible": in_thread,
            "search_visible": in_search,
            "notification_visible": notified,
            "reply_count_updated": reply_count_includes,
            "latest_reply_updated": latest_reply_is,
        }
        return Visibility(
            message_id=message_id,
            history_visible=in_history,
            thread_visible=in_thread,
            search_visible=in_search,
            notification_visible=notified,
            reply_count_includes=reply_count_includes,
            latest_reply_is=latest_reply_is,
            body=body,
            author=author,
            surfaces=surfaces,
        )

    def _channel_page(self, channel: str, actor: str) -> list[dict[str, Any]]:
        try:
            return execute_tool(self.db, "get_channel_messages", {"channel_id": channel}, actor)[
                "messages"
            ]
        except Exception:  # noqa: BLE001
            return []


# ---------------------------------------------------------------------------
# Invariants shared by every latent event
# ---------------------------------------------------------------------------


def assert_completely_hidden(visibility: Visibility) -> None:
    """A pending event must not leak through any Slack-facing surface."""
    leaked = [name for name, value in visibility.surfaces.items() if value]
    assert not leaked, (
        f"{visibility.message_id} is pending but leaked through {leaked}\n{visibility.render()}"
    )


def assert_observable_everywhere(
    visibility: Visibility, *, threaded: bool = True, expect_notification: bool = True
) -> None:
    """An activated event must be reachable wherever an ordinary message is.

    "Wherever an ordinary message is" depends on its shape. `get_channel_messages`
    returns top-level messages only -- documented in its schema, and true of the
    agent's own replies too -- so a threaded release is expected to be absent
    from channel history and present in `get_thread_replies`. Requiring history
    of a threaded message would report a propagation bug that is not there.

    Search is asserted separately by the caller, because whether a term matches
    is a property of the query rather than of activation.
    """
    required = ["thread_visible", "reply_count_updated"] if threaded else ["history_visible"]
    if expect_notification:
        required.append("notification_visible")
    missing = [name for name in required if not visibility.surfaces.get(name)]
    assert not missing, (
        f"{visibility.message_id} activated but is not observable via {missing}\n"
        f"{visibility.render()}"
    )


# ---------------------------------------------------------------------------
# Human-readable audit
# ---------------------------------------------------------------------------

#: The generic engine numbers keyword groups; a reader wants to know what the
#: group is *for*. These labels are presentation only -- matching never sees
#: them, so a mislabelled group cannot change whether an event fires.
PREDICATE_LABELS: dict[str, dict[str, str]] = {
    "r040_timing_confirmed": {
        "keyword_group_1": "contains_time_reference",
        "keyword_group_2": "contains_clarification_keyword",
    },
    "r010_saml_critique": {
        "keyword_group_1": "names_the_defective_comparison",
        "keyword_group_2": "names_the_required_comparison",
    },
    "r020_permissions_critique": {
        "keyword_group_1": "names_any_match_semantics",
        "keyword_group_2": "names_all_match_semantics",
    },
    "r050_rollback_checked": {
        "keyword_group_1": "names_rollback",
        "keyword_group_2": "asks_for_verification",
    },
    "r060_rehearsal_closed": {
        "keyword_group_1": "names_rollback",
        "keyword_group_2": "names_the_config_pin",
        "keyword_group_3": "asks_for_a_rerun",
    },
}


def label_for(rule_id: str, predicate_name: str) -> str:
    return PREDICATE_LABELS.get(rule_id, {}).get(predicate_name, predicate_name)


def render_evaluation(evaluation: RuleEvaluation) -> str:
    """The evaluator's own block, with semantic names for keyword groups."""
    text = evaluation.render()
    for raw, friendly in PREDICATE_LABELS.get(evaluation.rule_id, {}).items():
        text = text.replace(f"  {raw}:", f"  {friendly}:")
    return text


def audit(
    workspace: Workspace,
    event_id: str,
    rule_id: str,
    action: Action,
    message_id: str,
    channel: str,
    thread_parent: str | None = None,
    search_terms: str = "",
    notification_id: str | None = None,
) -> Outcome:
    """Run one action against one event and print a full causal account.

    This is the helper to reach for when a trajectory failed and the question
    is *why*: it prints the state before, the action, every predicate the rule
    weighed, whether the event moved, and what became observable afterwards.
    """
    before_status = workspace.event_status(event_id)
    before = workspace.probe(message_id, channel, thread_parent, search_terms, notification_id)

    print(f"EVENT AUDIT: {event_id}")
    print("\nInitial:")
    print(f"  status: {before_status}")
    print(f"  {message_id} visible: {str(before.anywhere).lower()}")

    print("\nAction:")
    print(f"  actor: {action.actor}")
    print(f"  tool: {action.tool}")
    for key in ("channel_id", "recipient_id", "thread_parent_id"):
        if action.payload.get(key):
            print(f"  {key}: {action.payload[key]}")
    if action.payload.get("body"):
        print(f'  body: "{action.payload["body"]}"')

    outcome = workspace.act(action)

    print("\nEvaluation:")
    evaluation = outcome.evaluation(rule_id)
    if evaluation is None:
        print(f"  rule {rule_id} was NOT CONSIDERED for this action")
        print("    (its event is no longer pending, or no rule by that id exists)")
    else:
        print(f"  rule {rule_id}")
        for predicate in evaluation.predicates:
            name = label_for(rule_id, predicate.name)
            verdict = "PASS" if predicate.passed else "FAIL"
            detail = ""
            if "candidates" in predicate.detail:
                detail = (f"  candidates={predicate.detail['candidates']} "
                          f"matched={predicate.detail.get('matched')}")
                if "matched_in_action_body" in predicate.detail:
                    detail += (
                        f"  [{predicate.detail['note']}; the action's own body "
                        f"contains {predicate.detail['matched_in_action_body']}]"
                    )
            elif "expected_thread" in predicate.detail:
                detail = (f"  expected={predicate.detail['expected_thread']} "
                          f"actual={predicate.detail.get('actual_thread')} "
                          f"conversation={predicate.detail.get('actual_conversation')}")
            elif "expected" in predicate.detail:
                detail = (f"  expected={predicate.detail['expected']!r} "
                          f"actual={predicate.detail.get('actual')!r}")
            print(f"    {name}: {verdict}{detail}")

    if outcome.error is not None:
        print(f"\n  tool refused: {type(outcome.error).__name__}: {outcome.error}")

    print("\nResult:")
    print(f"  activated: {str(bool(outcome.activated)).lower()} {outcome.activated or ''}")
    for activation in outcome.trace.activations:
        print("   ", activation.render().replace("\n", "\n    "))

    after = workspace.probe(message_id, channel, thread_parent, search_terms, notification_id)
    print("\nObservable afterward:")
    for name, value in after.surfaces.items():
        print(f"  {message_id} {name}: {str(value).lower()}")
    if after.body:
        print(f"  author: {after.author}")
        print(f"  body: {after.body[:110]}")
    print()
    return outcome
