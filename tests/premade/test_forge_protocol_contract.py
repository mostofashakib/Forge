"""Every premade app serves the same Forge protocol: health, reset, snapshots, restore."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# One state-changing call per app, so a test can tell a restore from a no-op.
_ACTIONS = {
    "gmail": ("/star", {"email_id": "e001"}),
    "slack": ("/send_message", {"channel": "general", "text": "hello"}),
}


@pytest.fixture(params=["gmail", "slack"])
def app(request, load_premade):
    client = TestClient(load_premade(request.param).app)
    client.post("/forge/reset")
    client.act = lambda: client.post(_ACTIONS[request.param][0], json=_ACTIONS[request.param][1])
    return client


def test_health_reports_ok(app):
    assert app.get("/forge/health").json() == {"status": "ok"}


def test_reset_returns_the_seeded_state(app):
    seeded = app.get("/forge/state").json()
    app.act()

    body = app.post("/forge/reset").json()

    assert body["status"] == "reset"
    assert body["state"] == seeded


def test_seeding_saves_a_baseline_snapshot(app):
    seeded = app.get("/forge/state").json()
    app.act()

    body = app.post("/forge/restore/baseline").json()

    assert body == {"status": "restored", "slot": "baseline", "state": seeded}


def test_a_snapshot_restores_the_state_it_saved(app):
    assert app.post("/forge/snapshot", json={"slot": "s1"}).json() == {"status": "snapshot_saved", "slot": "s1"}
    saved = app.get("/forge/state").json()
    app.act()
    assert app.get("/forge/state").json() != saved

    assert app.post("/forge/restore/s1").json()["state"] == saved


def test_saving_a_slot_again_overwrites_it(app):
    app.post("/forge/snapshot", json={"slot": "s1"})
    app.act()
    app.post("/forge/snapshot", json={"slot": "s1"})
    latest = app.get("/forge/state").json()

    assert app.post("/forge/restore/s1").json()["state"] == latest


def test_an_unknown_slot_is_not_found(app):
    response = app.post("/forge/restore/missing")

    assert response.status_code == 404
    assert "missing" in response.json()["detail"]


def test_reset_discards_saved_snapshots(app):
    app.post("/forge/snapshot", json={"slot": "s1"})

    app.post("/forge/reset")

    assert app.post("/forge/restore/s1").status_code == 404


def test_restore_state_returns_the_restored_view(app):
    dump = app.get("/forge/dump").json()
    view = app.get("/forge/state").json()
    app.act()

    body = app.post("/forge/restore-state", json=dump).json()

    assert body == {"status": "restored", "state": view}
