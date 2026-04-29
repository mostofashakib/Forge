from typing import TypedDict


class User(TypedDict):
    id: str
    email: str
    role: str


class Email(TypedDict):
    id: str
    from_: str
    to: str
    subject: str
    body: str
    labels: list[str]
    archived: bool
    thread_id: str
    read: bool
    created_at: str
    escalated: bool


class Thread(TypedDict):
    id: str
    email_ids: list[str]
    escalated: bool


class Label(TypedDict):
    id: str
    name: str


class GmailState(TypedDict):
    users: dict[str, User]
    emails: dict[str, Email]
    threads: dict[str, Thread]
    labels: dict[str, Label]
    actor_id: str
