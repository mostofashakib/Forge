"""The app runner drives a container app through its HTTP control plane.

A fake app behind httpx.MockTransport stands in for the container: reset,
snapshot, restore, restore-state, state, an optional dump, and actions.
"""
from __future__ import annotations

import copy
import json

import httpx
import pytest

from forge.taskfactory.pass_k import run_pass_k
from forge.taskfactory.runner import SeedError
from forge.taskfactory.runners.container_app import ContainerAppRunner
from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft, TaskSeed

BASELINE = {"emails": [{"id": "e1", "folder": "inbox"}], "counter": 1}


class FakeApp:
    def __init__(self, *, has_dump: bool = True) -> None:
        self.full = copy.deepcopy(BASELINE)
        self.slots: dict[str, dict] = {}
        self.has_dump = has_dump
        self.calls: list[str] = []

    def view(self) -> dict:
        # Like premade Gmail: the observed state is a view, not the full dump.
        return {"inbox": [e for e in self.full["emails"] if e["folder"] == "inbox"], "counter": self.full["counter"]}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        body = json.loads(request.content) if request.content else {}
        if path == "/forge/reset":
            # Like premade Gmail, reset also clears saved snapshot slots.
            self.full = copy.deepcopy(BASELINE)
            self.slots.clear()
            return httpx.Response(200, json={"ok": True})
        if path == "/forge/dump":
            return httpx.Response(200, json=self.full) if self.has_dump else httpx.Response(404)
        if path == "/forge/state":
            return httpx.Response(200, json=self.view() if self.has_dump else self.full)
        if path == "/forge/restore-state":
            self.full = body
            return httpx.Response(200, json={"ok": True})
        if path == "/forge/snapshot":
            self.slots[body["slot"]] = copy.deepcopy(self.full)
            return httpx.Response(200, json={"ok": True})
        if path.startswith("/forge/restore/"):
            self.full = copy.deepcopy(self.slots[path.rsplit("/", 1)[1]])
            return httpx.Response(200, json={"ok": True})
        if path == "/openapi.json":
            return httpx.Response(200, json={"paths": {
                "/archive_email": {"post": {"summary": "Archive an email", "requestBody": {"content": {
                    "application/json": {"schema": {"properties": {"email_id": {"type": "string"}}}}}}}},
                "/forge/reset": {"post": {}},
            }})
        if path == "/archive_email":
            email = next((e for e in self.full["emails"] if e["id"] == body.get("email_id")), None)
            if email is None:
                return httpx.Response(200, json={"ok": False, "error": "no such email"})
            email["folder"] = "archive"
            self.full["counter"] += 1
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)


def _runner(app: FakeApp) -> ContainerAppRunner:
    return ContainerAppRunner(httpx.Client(transport=httpx.MockTransport(app.handler), base_url="http://app"))


DRAFT = TaskDraft(
    slot=0, title="Archive e2", objective="Archive the email from finance.",
    seed=TaskSeed(records={"emails": [{"id": "e2", "folder": "inbox"}]}),
    golden=[GoldenStep(tool="/archive_email", args={"email_id": "e2"})],
    checks=[Check(kind="record_exists", collection="emails", match={"id": "e2"}, expect={"folder": "archive"})],
)


def test_a_sound_task_passes_k_runs():
    app = FakeApp()
    runner = _runner(app)

    with runner.batch():
        result = run_pass_k(runner, DRAFT, k=3)

    assert result.passed, result.reason


def test_the_seed_is_merged_into_the_full_dump_not_the_view():
    app = FakeApp()

    with _runner(app).session(DRAFT):
        pass

    assert {"id": "e2", "folder": "inbox"} in app.full["emails"]
    assert "inbox" not in app.full


def test_without_a_dump_endpoint_the_seed_merges_into_the_state():
    app = FakeApp(has_dump=False)

    with _runner(app).session(DRAFT):
        pass

    assert {"id": "e2", "folder": "inbox"} in app.full["emails"]


def test_an_action_the_app_rejects_in_its_body_is_a_failed_step():
    with _runner(FakeApp()).session(DRAFT) as session:
        result = session.step(GoldenStep(tool="/archive_email", args={"email_id": "nope"}))

    assert not result.ok
    assert "no such email" in result.error


def test_a_control_plane_step_never_reaches_the_app():
    app = FakeApp()

    with _runner(app).session(DRAFT) as session:
        result = session.step(GoldenStep(tool="/forge/restore-state", args={}))

    assert not result.ok
    assert app.calls.count("POST /forge/restore-state") == 1  # only the seed


def test_the_batch_puts_the_apps_state_back_afterwards():
    app = FakeApp()
    app.full["emails"].append({"id": "user-work", "folder": "inbox"})
    before = copy.deepcopy(app.full)

    runner = _runner(app)
    with runner.batch():
        run_pass_k(runner, DRAFT, k=2)

    assert app.full == before


def test_a_seed_into_a_missing_collection_raises_seed_error():
    draft = DRAFT.model_copy(update={"seed": TaskSeed(records={"calendar": [{"id": "c1"}]})})

    with pytest.raises(SeedError):
        with _runner(FakeApp()).session(draft):
            pass


def test_checks_read_the_full_state_not_the_view():
    # Archiving removes e2 from the inbox view. Only the full state still
    # shows where it went, so that is what checks read.
    check = Check(kind="record_exists", collection="emails", match={"id": "e2"}, expect={"folder": "archive"})

    with _runner(FakeApp()).session(DRAFT) as session:
        session.step(GoldenStep(tool="/archive_email", args={"email_id": "e2"}))
        outcome = session.evaluate([check])

    assert outcome[0].passed, outcome[0].detail


def test_the_profile_lists_actions_without_the_control_plane_and_samples_the_reset_state():
    app = FakeApp()
    app.full["emails"].append({"id": "user-work", "folder": "inbox"})

    profile = _runner(app).profile("mail", "premade:gmail")

    assert profile.tool_names == ["/archive_email"]
    assert profile.tools[0].params == ("email_id",)
    assert profile.state_sample == BASELINE
    assert {"id": "user-work", "folder": "inbox"} in app.full["emails"]
