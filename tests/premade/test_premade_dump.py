"""Premade apps expose their full restorable state at GET /forge/dump.

Their /forge/state is a view (an inbox, unread counts), while
/forge/restore-state takes the full database. The task factory seeds tasks
by merging rows into the dump and restoring it, so the dump must round-trip.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from forge.runtime.tools import OpenAPIToolProvider


@pytest.fixture(params=["gmail", "slack"])
def client(request, load_premade) -> TestClient:
    client = TestClient(load_premade(request.param).app)
    client.post("/forge/reset")
    return client


def test_restoring_the_dump_leaves_the_state_unchanged(client):
    before = client.get("/forge/state").json()

    client.post("/forge/restore-state", json=client.get("/forge/dump").json())

    assert client.get("/forge/state").json() == before


def test_the_dump_carries_the_virtual_clock_and_counters(client):
    assert "forge_counters" in client.get("/forge/dump").json()


def test_the_dump_reflects_changes_after_reset(client):
    baseline = client.get("/forge/dump").json()
    first_collection = next(key for key, value in baseline.items() if isinstance(value, list) and value)
    restored = {**baseline, first_collection: baseline[first_collection][1:]}

    client.post("/forge/restore-state", json=restored)

    assert client.get("/forge/dump").json() != baseline


def test_the_dump_is_not_an_agent_action(client):
    endpoints = [a["endpoint"] for a in OpenAPIToolProvider(client).action_manifest()]

    assert "/forge/dump" not in endpoints
