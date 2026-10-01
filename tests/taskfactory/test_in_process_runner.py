"""The in-process runner, against a real environment built with EnvBuilder."""
from __future__ import annotations

import copy

import pytest

from forge.contracts import InitialStateProvider
from forge.runtime.env_builder import EnvBuilder
from forge.runtime.errors import InvalidActionError
from forge.runtime.transition import TransitionResult
from forge.taskfactory.pass_k import run_pass_k
from forge.taskfactory.runner import SeedError
from forge.taskfactory.runners.in_process import InProcessRunner
from forge.taskfactory.schemas import Check, GoldenStep, TaskDraft, TaskSeed


class _Inbox(InitialStateProvider):
    def reset(self, ctx, *, seed, options):
        return {"emails": {"e1": {"id": "e1", "folder": "inbox"}}}


def _archive(state, action, ctx):
    email_id = action.get("email_id")
    if email_id not in state["emails"]:
        raise InvalidActionError(f"no email {email_id}")
    new_state = copy.deepcopy(state)
    new_state["emails"][email_id]["folder"] = "archive"
    return TransitionResult(state=new_state, events=[{"type": "archived", "entity_id": email_id}])


def _build(max_steps: int):
    return (
        EnvBuilder("inbox", domain="mail", max_steps=max_steps)
        .with_initial_state(_Inbox())
        .with_transition("archive", _archive)
        .build(verify=False)
    )


def _draft(**overrides) -> TaskDraft:
    fields = dict(
        slot=0, title="Archive the invoice", objective="Archive email e2.",
        seed=TaskSeed(records={"emails": [{"id": "e2", "folder": "inbox"}]}),
        golden=[GoldenStep(tool="archive", args={"email_id": "e2"})],
        checks=[
            Check(kind="record_exists", collection="emails", match={"id": "e2"}, expect={"folder": "archive"}),
            Check(kind="actions_called", tools=["archive"]),
        ],
    )
    fields.update(overrides)
    return TaskDraft(**fields)


def test_a_sound_task_passes_k_runs_with_a_stable_fingerprint():
    result = run_pass_k(InProcessRunner(_build), _draft(), k=3)

    assert result.passed, result.reason
    assert len(result.fingerprint) == 64


def test_the_seed_is_merged_before_the_task_starts():
    runner = InProcessRunner(_build)

    with runner.session(_draft()) as session:
        outcome = session.evaluate([Check(kind="record_exists", collection="emails", match={"id": "e2"})])

    assert outcome[0].passed


def test_a_golden_step_the_environment_refuses_is_reported():
    draft = _draft(golden=[GoldenStep(tool="archive", args={"email_id": "e9"})])

    result = run_pass_k(InProcessRunner(_build), draft, k=2)

    assert not result.passed
    assert "e9" in result.reason


def test_an_unknown_action_is_reported_as_a_failed_step():
    with InProcessRunner(_build).session(_draft()) as session:
        result = session.step(GoldenStep(tool="explode"))

    assert not result.ok


def test_a_seed_into_a_missing_collection_raises_seed_error():
    draft = _draft(seed=TaskSeed(records={"calendar": [{"id": "c1"}]}))

    with pytest.raises(SeedError, match="calendar"):
        with InProcessRunner(_build).session(draft):
            pass


def test_the_environment_is_built_with_room_for_the_whole_golden_solution():
    sizes: list[int] = []

    def build(max_steps):
        sizes.append(max_steps)
        return _build(max_steps)

    long = _draft(golden=[GoldenStep(tool="archive", args={"email_id": "e2"})] * 40)
    with InProcessRunner(build).session(long):
        pass

    assert sizes[0] > 40
