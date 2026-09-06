"""What a task adds to the Slack simulator.

The simulator itself is a Slack workspace and nothing else: channels, threads,
membership, search, a clock. It has no idea which task it is hosting.

A scenario is the task-shaped half, and it is *data*. It says which things can
happen in this world (`events`), what other people say when they do
(`latent_messages`), and what the agent has to do for them to happen (`rules`).
The tracker in `tracker` evaluates those rules generically, so
adding a second task means writing a second `Scenario`, not editing the
environment.

An empty `Scenario()` is legal and gives a plain, static Slack workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slack_sim.models import (
    LatentMembership,
    LatentMessage,
    LatentNotification,
    ScenarioEvent,
    ScenarioRule,
)


@dataclass(frozen=True, slots=True)
class Scenario:
    events: tuple[ScenarioEvent, ...] = ()
    latent_messages: tuple[LatentMessage, ...] = ()
    #: Inbox entries and memberships that appear with their event, so a task
    #: can stage more than conversation: who gets told, and who joins.
    latent_notifications: tuple[LatentNotification, ...] = ()
    latent_memberships: tuple[LatentMembership, ...] = ()
    rules: tuple[ScenarioRule, ...] = ()
    #: Ids of events that carry no messages of their own. Recording that the
    #: agent got somewhere is often the point, so this is not an edge case.
    observation_only: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        declared = {event.event_id for event in self.events}
        for message in self.latent_messages:
            if message.event_id not in declared:
                raise ValueError(
                    f"latent message {message.message_id} belongs to undeclared "
                    f"event {message.event_id}"
                )
        for notification in self.latent_notifications:
            if notification.event_id not in declared:
                raise ValueError(
                    f"latent notification {notification.notification_id} belongs to "
                    f"undeclared event {notification.event_id}"
                )
        for membership in self.latent_memberships:
            if membership.event_id not in declared:
                raise ValueError(
                    f"latent membership {membership.membership_id} belongs to "
                    f"undeclared event {membership.event_id}"
                )
        for rule in self.rules:
            unknown = ({rule.event_id} | set(rule.requires_activated)
                       | set(rule.requires_pending)) - declared
            if unknown:
                raise ValueError(f"rule {rule.rule_id} references unknown events: {sorted(unknown)}")

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)
