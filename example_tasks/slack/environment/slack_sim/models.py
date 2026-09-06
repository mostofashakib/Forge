"""Typed domain records for seeded Slack state and relationships."""

from __future__ import annotations

import json
from dataclasses import astuple, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Record:
    def row(self) -> tuple[Any, ...]:
        return astuple(self)


@dataclass(frozen=True, slots=True)
class SlackUser(Record):
    user_id: str
    display_name: str
    email: str
    role: str
    team: str
    handle: str


@dataclass(frozen=True, slots=True)
class UserProfile(Record):
    user_id: str
    manager_id: str | None
    title: str
    timezone: str
    status_text: str


@dataclass(frozen=True, slots=True)
class Channel(Record):
    channel_id: str
    name: str
    is_private: bool
    owner_id: str
    created_step: int = 0


@dataclass(frozen=True, slots=True)
class ChannelTopic(Record):
    channel_id: str
    topic: str


@dataclass(frozen=True, slots=True)
class Membership(Record):
    membership_id: str
    channel_id: str
    user_id: str
    role: str
    joined_step: int = 0


@dataclass(frozen=True, slots=True)
class Message(Record):
    message_id: str
    conversation_id: str
    author_id: str
    text: str
    created_step: int
    #: The thread this message belongs to: the root's ID, or None for a root.
    thread_parent_id: str | None = None
    edited_step: int | None = None
    #: The specific message this one answers. Equal to `thread_parent_id` when
    #: the reply is addressed to the thread root, and pointing at a sibling
    #: reply when it answers something said inside the thread.
    reply_to_id: str | None = None


@dataclass(frozen=True, slots=True)
class LatentMessage(Record):
    """A deterministic Slack message that is invisible until its event fires."""

    message_id: str
    event_id: str
    conversation_id: str
    author_id: str
    text: str
    thread_parent_id: str | None = None
    reply_to_id: str | None = None
    ordinal: int = 0


@dataclass(frozen=True, slots=True)
class Pin(Record):
    """A pinned message. Pins belong to the conversation, not to one person."""

    conversation_id: str
    message_id: str
    user_id: str
    pinned_step: int = 0


@dataclass(frozen=True, slots=True)
class SavedItem(Record):
    """A personal bookmark, visible only to the person who saved it."""

    user_id: str
    message_id: str
    saved_step: int = 0


@dataclass(frozen=True, slots=True)
class ThreadFollow(Record):
    """A subscription to a thread's later replies."""

    user_id: str
    thread_id: str
    followed_step: int = 0


@dataclass(frozen=True, slots=True)
class LatentNotification(Record):
    """An inbox entry that appears when its event fires.

    A future message usually deserves a future notification, and deriving one
    from the other would make the inbox depend on who happened to be following
    what at activation time. Declaring it keeps the episode reproducible.
    """

    notification_id: str
    event_id: str
    user_id: str
    kind: str
    message_id: str
    conversation_id: str


@dataclass(frozen=True, slots=True)
class LatentMembership(Record):
    """A membership that appears when its event fires.

    Somebody joining a channel partway through is ordinary workplace
    behaviour, and it is state a reader can check, so it belongs in the same
    latent mechanism as messages rather than in bespoke code.
    """

    membership_id: str
    event_id: str
    channel_id: str
    user_id: str
    role: str = "member"


@dataclass(frozen=True, slots=True)
class Notification(Record):
    """One entry in somebody's inbox, as part of the declared initial state.

    Seeded notifications are data like every other seeded record, so the
    workspace a run starts from is identical every time and can be read
    straight out of the fixture rather than recomputed from it.
    `read_step` is None when the notification is still unread.
    """

    notification_id: str
    user_id: str
    kind: str
    message_id: str
    conversation_id: str
    created_step: int
    read_step: int | None = None

    KINDS = ("mention", "direct_message", "thread_reply", "reaction")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(f"{self.notification_id}: unknown kind {self.kind!r}")


@dataclass(frozen=True, slots=True)
class ScenarioEvent(Record):
    """One thing that can happen in the world once, if its rules are satisfied.

    `scheduled_step` pins the event to an instant on the seed calendar instead
    of letting it land one second after whatever the agent just did. Use it for
    things the world does on its own clock -- a nightly job, a meeting -- and
    leave it None for everything that happens in response to the agent.
    """

    event_id: str
    scheduled_step: int | None = None


@dataclass(frozen=True, slots=True)
class ScenarioRule:
    """A condition under which an event activates.

    Rules are data, not code: the transition engine understands the four
    trigger kinds below and nothing about any particular scenario.

      reply_keywords  the actor addresses the rule (see below) with a body
                      containing at least one word from every group in
                      `keyword_groups`
      reply_exact     the actor replies in `thread_id` with exactly
                      `exact_body`, ignoring case and trailing punctuation
      observed        any id in `observed_ids` appeared in a result the actor
                      actually received
      channel_joined  the actor joined `channel_id`
      all_of          no trigger of its own; fires as soon as its
                      prerequisites hold

    Every kind is additionally gated by `requires_activated` and
    `requires_pending`, which is how a scenario expresses ordering: a review
    approval that must follow a critique, a re-run that must follow a fix.

    Addressing. A `reply_keywords` rule asks the agent to raise something with
    someone, and there is usually more than one defensible place to do that. A
    rule therefore declares where it will listen, and any one of them counts:

      thread_id     a reply in this thread -- the canonical place
      channel_id    any message the actor posts in this channel
      recipient_ids a direct message to any of these people

    `tools` names the operations that can carry it, defaulting to
    `reply_to_thread`. Widening *where* is deliberate; widening *what the agent
    must demonstrate* is not, so `keyword_groups` still has to match.
    `reply_exact` ignores all of this and stays anchored to its thread, because
    the task instructs a verbatim string.
    """

    rule_id: str
    event_id: str
    trigger: str
    thread_id: str | None = None
    channel_id: str | None = None
    exact_body: str | None = None
    keyword_groups: tuple[tuple[str, ...], ...] = ()
    observed_ids: tuple[str, ...] = ()
    requires_activated: tuple[str, ...] = ()
    requires_pending: tuple[str, ...] = ()
    recipient_ids: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()

    TRIGGERS = ("reply_keywords", "reply_exact", "observed", "channel_joined", "all_of")
    #: What carries an ask when a rule does not say otherwise.
    DEFAULT_TOOLS = ("reply_to_thread",)

    def __post_init__(self) -> None:
        if self.trigger not in self.TRIGGERS:
            raise ValueError(f"{self.rule_id}: unknown trigger {self.trigger!r}")
        if self.trigger == "reply_exact" and not self.thread_id:
            raise ValueError(f"{self.rule_id}: reply_exact needs a thread_id")
        if self.trigger == "reply_keywords" and not (
            self.thread_id or self.channel_id or self.recipient_ids
        ):
            raise ValueError(
                f"{self.rule_id}: reply_keywords needs somewhere to listen "
                "(thread_id, channel_id or recipient_ids)"
            )
        if self.trigger == "reply_exact" and not self.exact_body:
            raise ValueError(f"{self.rule_id}: reply_exact needs exact_body")
        if self.trigger == "reply_keywords" and not self.keyword_groups:
            raise ValueError(f"{self.rule_id}: reply_keywords needs keyword_groups")
        if self.trigger == "observed" and not self.observed_ids:
            raise ValueError(f"{self.rule_id}: observed needs observed_ids")
        if self.trigger == "channel_joined" and not self.channel_id:
            raise ValueError(f"{self.rule_id}: channel_joined needs channel_id")
        if self.trigger == "all_of" and not self.requires_activated:
            raise ValueError(f"{self.rule_id}: all_of needs requires_activated")

    def row(self) -> tuple[Any, ...]:
        """Flatten for SQLite; the list-valued fields travel as JSON."""
        return (
            self.rule_id, self.event_id, self.trigger, self.thread_id, self.channel_id,
            self.exact_body,
            json.dumps([list(group) for group in self.keyword_groups]),
            json.dumps(list(self.observed_ids)),
            json.dumps(list(self.requires_activated)),
            json.dumps(list(self.requires_pending)),
            json.dumps(list(self.recipient_ids)),
            json.dumps(list(self.tools)),
        )

    def accepted_tools(self) -> tuple[str, ...]:
        return self.tools or self.DEFAULT_TOOLS


@dataclass(frozen=True, slots=True)
class Reaction(Record):
    reaction_id: str
    message_id: str
    user_id: str
    emoji: str
    created_step: int = 0


@dataclass(frozen=True, slots=True)
class Chat(Record):
    chat_id: str
    kind: str
    name: str | None
    created_step: int


@dataclass(frozen=True, slots=True)
class ChatParticipant(Record):
    chat_id: str
    user_id: str


@dataclass(frozen=True, slots=True)
class UserGroup(Record):
    user_group_id: str
    name: str
    handle: str
    owner_id: str


@dataclass(frozen=True, slots=True)
class UserGroupMember(Record):
    user_group_id: str
    user_id: str


@dataclass(frozen=True, slots=True)
class MessageMention(Record):
    message_id: str
    mention_type: str
    target_id: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class ConversationRead(Record):
    conversation_id: str
    user_id: str
    last_read_step: int
