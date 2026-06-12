"""Canonical serialization and hashing — the single source for both.

Different libraries (and dict insertion orders) serialize the same object
differently; everything that hashes or compares state must go through here.
"""
from __future__ import annotations
import hashlib
import json


def canonical_dumps(obj) -> str:
    """Serialize with sorted keys and fixed separators so the same object
    always produces the same bytes regardless of insertion order or library."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def canonical_hash(obj) -> str:
    return hashlib.sha256(canonical_dumps(obj).encode()).hexdigest()
