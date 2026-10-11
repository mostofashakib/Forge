"""Detectors for credentials. A secret is redacted outright, never pseudonymized.

Patterns need a provider prefix plus enough length to be a real key, so ids
like `task-sk-ticket` or a bare `Bearer` in prose pass through.
"""
from __future__ import annotations

import re

from forge.safety.types import Span

REDACTED_SECRET = "[REDACTED_SECRET]"

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?<![\w-])sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    # Real bearer tokens are long. The shorter logging redactor would flag prose.
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    # user:password in a URL authority.
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),
]


def detect_secrets(text: str) -> list[Span]:
    """Every credential in `text`, in no particular order."""
    return [
        Span(match.start(), match.end(), "secret", match.group())
        for pattern in _PATTERNS
        for match in pattern.finditer(text)
    ]
