"""The pipeline: taxonomy, writer, three validation layers, and top-up rounds.

Rejected slots go back to the writer with their reason, for up to three
rounds. A batch that still falls short is returned short, never padded, and
every rejection is kept.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from forge.taskfactory.pass_k import PassKResult
from forge.taskfactory.pipeline import TaskFactoryPipeline
from forge.taskfactory.profile import EnvironmentProfile, ToolInfo
from forge.taskfactory.schemas import (
    Check,
    GoldenStep,
    ReviewVerdict,
    TaskDraft,
    Taxonomy,
    TaxonomyCategory,
)

PROFILE = EnvironmentProfile(
    env_name="mail", env_type="premade:gmail", family="state",
    tools=(ToolInfo(name="/archive_email"),), state_sample={"emails": []},
)
TAXONOMY = Taxonomy(
    categories=[TaxonomyCategory(name="triage", description="sort", exercises=[], difficulties=[1])],
    difficulty_rubric={},
)


class _Builder:
    def build(self, profile, count):
        return TAXONOMY


def _draft(slot: int, *, tool: str = "/archive_email", title: str | None = None) -> TaskDraft:
    return TaskDraft(
        slot=slot, title=title or f"task {slot}", objective="archive e1",
        golden=[GoldenStep(tool=tool, args={"email_id": "e1"})],
        checks=[Check(kind="record_exists", collection="emails", match={"id": "e1"}, expect={"folder": "archive"})],
    )


class _Writer:
    """Writes per round from a script: {round: {slot: draft or None}}. Default: a good draft.

    `revisions` maps a slot to the draft `revise` returns for it.
    """

    def __init__(self, script: dict[int, dict[int, TaskDraft | None]] | None = None, revisions=None) -> None:
        self.script = script or {}
        self.revisions = revisions or {}
        self.calls: list[dict] = []
        self.revised: list[tuple[int, str]] = []

    def revise(self, profile, taxonomy, items):
        self.revised += [(slot.index, problem) for slot, _draft, problem in items]
        return {slot.index: self.revisions[slot.index] for slot, _d, _p in items if slot.index in self.revisions}

    def write(self, profile, taxonomy, slots, *, avoid_titles=(), feedback=None):
        round_number = len(self.calls) + 1
        self.calls.append({"slots": [s.index for s in slots], "feedback": dict(feedback or {}), "titles": list(avoid_titles)})
        planned = self.script.get(round_number, {})
        drafts = {}
        for slot in slots:
            draft = planned.get(slot.index, _draft(slot.index, title=f"task {slot.index} r{round_number}"))
            if draft is not None:
                drafts[slot.index] = draft
        return drafts


def _verdict(slot: int, ok: bool = True) -> ReviewVerdict:
    return ReviewVerdict(
        slot=slot, realistic=True, realistic_reason="yes", fair=ok,
        fair_reason="fine" if ok else "hidden requirement", sensible=True, sensible_reason="yes",
    )


class _Reviewer:
    def __init__(self, reject: set[tuple[int, int]] = frozenset()) -> None:
        self.reject = reject
        self.seen: list[list[int]] = []

    def review(self, profile, tasks):
        round_number = len(self.seen) + 1
        self.seen.append([slot.index for slot, _ in tasks])
        return {slot.index: _verdict(slot.index, (round_number, slot.index) not in self.reject) for slot, _ in tasks}


class _Runner:
    def __init__(self) -> None:
        self.batches = 0

    @contextmanager
    def batch(self):
        self.batches += 1
        yield


def _pipeline(writer=None, reviewer=None, pass_k=None, runner=None, exclusive=None, progress=None):
    return TaskFactoryPipeline(
        profile=PROFILE,
        taxonomy_builder=_Builder(),
        writer=writer or _Writer(),
        reviewer=reviewer or _Reviewer(),
        runner=runner or _Runner(),
        pass_k=pass_k or (lambda runner, draft, k: PassKResult(True, fingerprint="fp")),
        exclusive=exclusive,
        progress=progress,
    )


def test_a_clean_batch_delivers_exactly_the_requested_tasks():
    result = _pipeline().run(count=4, k=3)

    assert result.status == "complete"
    assert [t.id for t in result.tasks] == ["t-00001", "t-00002", "t-00003", "t-00004"]
    assert all(t.step_budget == 2 for t in result.tasks)
    assert all(t.fingerprint == "fp" and t.round == 1 for t in result.tasks)
    assert result.rejections == []


def test_a_static_rejection_is_rewritten_with_its_reason():
    writer = _Writer({1: {1: _draft(1, tool="/nuke_inbox")}})

    result = _pipeline(writer=writer).run(count=3, k=1)

    assert result.status == "complete"
    assert writer.calls[1]["slots"] == [1]
    assert "/nuke_inbox" in writer.calls[1]["feedback"][1]
    assert [r.stage for r in result.rejections] == ["static"]
    assert next(t for t in result.tasks if t.id == "t-00002").round == 2


def test_rewrites_avoid_titles_already_accepted():
    writer = _Writer({1: {0: None}})

    _pipeline(writer=writer).run(count=2, k=1)

    assert "task 1 r1" in writer.calls[1]["titles"]


def test_a_pass_k_failure_never_reaches_the_reviewer():
    def pass_k(runner, draft, k):
        if draft.slot == 0 and draft.title.endswith("r1"):
            return PassKResult(False, "run 2: golden step 1 failed")
        return PassKResult(True, fingerprint="fp")

    reviewer = _Reviewer()
    result = _pipeline(reviewer=reviewer, pass_k=pass_k).run(count=2, k=3)

    assert reviewer.seen[0] == [1]
    assert result.rejections[0].stage == "pass_k"
    assert "golden step 1 failed" in result.rejections[0].reason
    assert result.status == "complete"


def test_a_review_rejection_keeps_the_validators_reason():
    result = _pipeline(reviewer=_Reviewer(reject={(1, 0)})).run(count=2, k=1)

    rejection = result.rejections[0]
    assert rejection.stage == "review"
    assert "hidden requirement" in rejection.reason


def test_a_slot_that_keeps_failing_leaves_the_batch_short_after_three_rounds():
    writer = _Writer({r: {0: _draft(0, tool="/nuke_inbox")} for r in (1, 2, 3)})

    result = _pipeline(writer=writer).run(count=3, k=1)

    assert len(writer.calls) == 3
    assert result.status == "short"
    assert result.shortfall == 1
    assert len(result.tasks) == 2
    assert len(result.rejections) == 3


def test_a_slot_the_writer_skipped_is_recorded_and_retried():
    writer = _Writer({1: {2: None}})

    result = _pipeline(writer=writer).run(count=3, k=1)

    assert result.status == "complete"
    assert "no task" in writer.calls[1]["feedback"][2]


def test_execution_holds_the_environment_lock_once_per_round():
    entered: list[int] = []

    @contextmanager
    def exclusive():
        entered.append(1)
        yield

    runner = _Runner()
    writer = _Writer({1: {0: _draft(0, tool="/nuke_inbox")}})
    _pipeline(writer=writer, runner=runner, exclusive=exclusive).run(count=2, k=1)

    assert len(entered) == 2
    assert runner.batches == 2


def test_progress_reports_each_stage():
    events: list[dict] = []

    _pipeline(progress=events.append).run(count=1, k=1)

    stages = [e["stage"] for e in events]
    for stage in ("taxonomy", "writing", "static", "pass_k", "review"):
        assert stage in stages


def test_progress_reports_each_rejection_with_its_reason():
    events: list[dict] = []
    writer = _Writer({1: {0: _draft(0, tool="/nuke_inbox")}})

    _pipeline(writer=writer, progress=events.append).run(count=1, k=1)

    rejected = [e for e in events if e.get("rejected")]
    assert len(rejected) == 1
    assert "/nuke_inbox" in rejected[0]["log"]
    assert rejected[0]["stage"] == "static"


def _short(slot: int, steps: int) -> TaskDraft:
    return _draft(slot).model_copy(update={"golden": [GoldenStep(tool="/archive_email")] * steps})


def test_a_draft_off_only_on_length_is_revised_in_the_same_round():
    # Slot 0 is difficulty 1 in this taxonomy; a 6-step golden is too long,
    # and the revision brings it back inside the range.
    writer = _Writer({1: {0: _short(0, 6)}}, revisions={0: _draft(0, title="revised")})

    result = _pipeline(writer=writer).run(count=1, k=1)

    assert result.status == "complete"
    assert len(writer.calls) == 1
    assert writer.revised[0][0] == 0 and "1 to 5" in writer.revised[0][1]
    assert result.tasks[0].round == 1 and result.tasks[0].title == "revised"


def test_a_draft_with_other_problems_is_not_revised():
    writer = _Writer({1: {0: _draft(0, tool="/nuke_inbox")}})

    _pipeline(writer=writer).run(count=1, k=1)

    assert writer.revised == []


def test_a_revision_still_off_on_length_is_rejected_and_rewritten_next_round():
    writer = _Writer({1: {0: _short(0, 6)}}, revisions={0: _short(0, 7)})

    result = _pipeline(writer=writer).run(count=1, k=1)

    assert len(writer.calls) == 2
    assert result.rejections[0].stage == "static"
    assert "got 7" in result.rejections[0].reason


def test_a_count_outside_one_to_twenty_thousand_is_rejected():
    with pytest.raises(ValueError):
        _pipeline().run(count=20_001, k=1)
    with pytest.raises(ValueError):
        _pipeline().run(count=0, k=1)


def test_a_round_is_processed_in_chunks_each_under_its_own_lock():
    # Agent runs on the environment get a turn between chunks, instead of
    # waiting for a whole round of a large batch.
    entered: list[int] = []

    @contextmanager
    def exclusive():
        entered.append(1)
        yield

    writer = _Writer()
    result = _pipeline(writer=writer, exclusive=exclusive).run(count=25, k=1)

    assert result.status == "complete"
    assert [len(call["slots"]) for call in writer.calls] == [10, 10, 5]
    assert len(entered) == 3


def test_a_draft_repeating_an_accepted_title_is_rejected():
    writer = _Writer({1: {1: _draft(1, title="task 0 r1")}})

    result = _pipeline(writer=writer).run(count=2, k=1)

    assert any("duplicate" in r.reason for r in result.rejections)


def test_the_writer_sees_a_bounded_list_of_titles_from_the_same_category():
    writer = _Writer()

    _pipeline(writer=writer).run(count=200, k=1)

    assert max(len(call["titles"]) for call in writer.calls) <= 50


def test_a_k_outside_one_to_ten_is_rejected():
    with pytest.raises(ValueError):
        _pipeline().run(count=1, k=0)


class _EdgeWriter(_Writer):
    """Writes drafts that plant each slot's edge case in a seeded email."""

    def write(self, profile, taxonomy, slots, *, avoid_titles=(), feedback=None):
        self.calls.append({"slots": [s.index for s in slots], "edges": [s.edge_case for s in slots]})
        return {slot.index: _edge_draft(slot.index, slot.edge_case) for slot in slots}


def _edge_draft(slot: int, edge_case: str | None) -> TaskDraft:
    from forge.taskfactory.schemas import TaskSeed

    return _draft(slot).model_copy(update={
        "seed": TaskSeed(records={"emails": [{"id": "e1", "folder": "inbox"}]}),
        "edge_case": edge_case,
        "edge_case_seed": ["emails:e1"] if edge_case else [],
    })


def test_requested_edge_cases_reach_the_writer_and_the_accepted_tasks():
    writer = _EdgeWriter()

    result = _pipeline(writer=writer).run(count=6, k=1, edge_cases=("recovery",))

    assert result.status == "complete"
    edges = [task.edge_case for task in result.tasks]
    assert edges.count("recovery") == 3 and edges.count(None) == 3
    assert writer.calls[0]["edges"].count("recovery") == 3
    assert all(task.edge_case_seed == ["emails:e1"] for task in result.tasks if task.edge_case)
    assert all(task.edge_case_seed == [] for task in result.tasks if task.edge_case is None)


class _CountingBuilder(_Builder):
    def __init__(self) -> None:
        self.calls = 0

    def build(self, profile, count):
        self.calls += 1
        return TAXONOMY


def _browser_pipeline(builder) -> TaskFactoryPipeline:
    return TaskFactoryPipeline(
        profile=EnvironmentProfile(env_name="web", env_type="browser", family="browser"),
        taxonomy_builder=builder, writer=_Writer(), reviewer=_Reviewer(), runner=_Runner(),
        pass_k=lambda runner, draft, k: PassKResult(True, fingerprint="fp"),
    )


def test_an_edge_case_the_family_does_not_offer_fails_before_any_llm_call():
    builder = _CountingBuilder()

    with pytest.raises(ValueError, match="missing_deps"):
        _browser_pipeline(builder).run(count=3, k=1, edge_cases=("missing_deps",))
    assert builder.calls == 0


def test_an_unknown_edge_case_fails_before_any_llm_call():
    builder = _CountingBuilder()

    with pytest.raises(ValueError, match="permission_errors"):
        _browser_pipeline(builder).run(count=3, k=1, edge_cases=("permission_errors",))
    assert builder.calls == 0


# ---------------------------------------------------------------------------
# Circuit breakers: a batch that stops producing tasks, or runs out of time,
# stops itself and keeps what it accepted.
# ---------------------------------------------------------------------------

def _all_bad(rounds: range, slots: range) -> dict[int, dict[int, TaskDraft]]:
    return {r: {s: _draft(s, tool="/nuke_inbox") for s in slots} for r in rounds}


def _stopped(result) -> list:
    return [r for r in result.rejections if r.stage == "stopped"]


def test_a_batch_that_accepts_nothing_for_three_chunks_in_a_row_stops():
    writer = _Writer(_all_bad(range(1, 10), range(40)))
    events: list[dict] = []

    result = _pipeline(writer=writer, progress=events.append).run(count=40, k=1)

    assert len(writer.calls) == 3
    assert result.status == "short"
    assert result.tasks == []
    assert sorted(r.slot for r in _stopped(result)) == list(range(30, 40))
    assert all("3 chunks in a row" in r.reason for r in _stopped(result))
    assert any(e["stage"] == "stopped" for e in events)


def test_a_chunk_that_accepts_a_task_resets_the_stall_count():
    script = _all_bad(range(1, 6), range(50))
    del script[3][20]  # the third chunk accepts slot 20

    writer = _Writer(script)
    result = _pipeline(writer=writer).run(count=50, k=1)

    assert _stopped(result) == []
    assert result.status == "complete"


def test_a_small_batch_still_gets_all_three_rounds():
    writer = _Writer(_all_bad(range(1, 3), range(5)))

    result = _pipeline(writer=writer).run(count=5, k=1)

    assert len(writer.calls) == 3
    assert result.status == "complete"
    assert _stopped(result) == []


class _Clock:
    def __init__(self, *times: float) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


def _timed_pipeline(writer, clock, time_limit_s):
    return TaskFactoryPipeline(
        profile=PROFILE, taxonomy_builder=_Builder(), writer=writer, reviewer=_Reviewer(), runner=_Runner(),
        pass_k=lambda runner, draft, k: PassKResult(True, fingerprint="fp"),
        time_limit_s=time_limit_s, clock=clock,
    )


def test_a_batch_past_its_time_limit_stops_before_the_next_chunk_and_keeps_its_tasks():
    writer = _Writer()
    # start, before chunk 1, before chunk 2
    result = _timed_pipeline(writer, _Clock(0, 50, 150), time_limit_s=100).run(count=25, k=1)

    assert len(writer.calls) == 1
    assert len(result.tasks) == 10
    assert result.status == "short"
    assert sorted(r.slot for r in _stopped(result)) == list(range(10, 25))
    assert all("time limit" in r.reason for r in _stopped(result))


def test_a_batch_inside_its_time_limit_is_never_stopped():
    result = _timed_pipeline(_Writer(), _Clock(0, 99), time_limit_s=100).run(count=25, k=1)

    assert result.status == "complete"
    assert _stopped(result) == []


def test_a_time_limit_that_is_not_positive_is_rejected():
    with pytest.raises(ValueError, match="time limit"):
        _timed_pipeline(_Writer(), _Clock(0), time_limit_s=0)
