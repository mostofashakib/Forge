"""Stable keyed placeholders, so one person stays one person across a dataset."""
from __future__ import annotations

import hashlib
import hmac
import re

_DIGEST_CHARS = 8
# Exactly the placeholders `replace` writes, and nothing shaped like them.
PSEUDONYM_PATTERN = re.compile(
    r"user_[0-9a-f]{8}@example\.invalid(?![\w.-])"
    r"|\b(?:Person|PHONE|SSN|CARD|IBAN|IP|ADDRESS)_[0-9a-f]{8}\b"
)
_DIGIT_KINDS = {"phone", "card", "ssn"}


class Pseudonymizer:
    def __init__(self, key: bytes) -> None:
        if not key:
            raise ValueError("Pseudonymizer key must not be empty")
        self._key = key

    def replace(self, kind: str, value: str) -> str:
        digest = hmac.new(
            self._key, f"{kind}:{_normalize(kind, value)}".encode(), hashlib.sha256
        ).hexdigest()[:_DIGEST_CHARS]
        if kind == "email":
            return f"user_{digest}@example.invalid"
        if kind == "person":
            return f"Person_{digest}"
        return f"{kind.upper()}_{digest}"


def _normalize(kind: str, value: str) -> str:
    """Two spellings of one value hash the same: case for text, digits for numbers."""
    if kind in _DIGIT_KINDS:
        return re.sub(r"\D", "", value)
    if kind in {"iban", "ip"}:
        return re.sub(r"\s", "", value).lower()
    return " ".join(value.split()).lower()
