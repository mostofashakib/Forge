# forge/runtime/http_state.py
"""State manager for container-backed environments.

The container app's SQLite database is the source of truth; this reads and
writes it through the Forge control endpoints rather than holding a copy.
"""
from __future__ import annotations

import hashlib
import json

import httpx

from forge.contracts import StateManager


class HttpStateManager(StateManager):
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=15.0)

    def get(self) -> dict:
        response = self._client.get(f"{self._base_url}/forge/state")
        response.raise_for_status()
        return response.json()

    def apply(self, new_state: dict) -> None:
        response = self._client.post(
            f"{self._base_url}/forge/restore-state", json=new_state
        )
        response.raise_for_status()

    def hash(self) -> str:
        serialized = json.dumps(self.get(), sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()}"

    def snapshot(self, slot: str) -> None:
        response = self._client.post(
            f"{self._base_url}/forge/snapshot", json={"slot": slot}
        )
        response.raise_for_status()

    def restore(self, slot: str) -> None:
        response = self._client.post(f"{self._base_url}/forge/restore/{slot}")
        response.raise_for_status()
