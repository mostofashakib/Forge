"""Canonical byte-stable serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain_data(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_plain_data(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, set):
        return [to_plain_data(item) for item in sorted(value)]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_plain_data(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")

