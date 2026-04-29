from typing import TypedDict


class ReplyEmailAction(TypedDict):
    type: str          # "reply_email"
    thread_id: str
    body: str


class SendEmailAction(TypedDict):
    type: str          # "send_email"
    to: str
    subject: str
    body: str


class ArchiveEmailAction(TypedDict):
    type: str          # "archive_email"
    email_id: str


class ApplyLabelAction(TypedDict):
    type: str          # "apply_label"
    email_id: str
    label: str


class MarkReadAction(TypedDict):
    type: str          # "mark_read"
    email_id: str


class EscalateThreadAction(TypedDict):
    type: str          # "escalate_thread"
    thread_id: str
