"""The writer fills slots in chunks, and the reviewer judges tasks in chunks.

Both are LLM calls behind a schema, so these tests use scripted clients and
check what is sent and how the answers map back to slots.
"""
from __future__ import annotations

from forge.taskfactory.profile import EnvironmentProfile, ToolInfo
from forge.taskfactory.review import TaskReviewer
from forge.taskfactory.schemas import (
    Check,
    GoldenStep,
    ReflectionPoint,
    ReviewVerdict,
    TaskDraft,
    TaskDraftBatch,
    Taxonomy,
    TaxonomyCategory,
)
from forge.taskfactory.slots import Slot
from forge.taskfactory.writer import TaskWriter

PROFILE = EnvironmentProfile(
    env_name="mail", env_type="premade:gmail", family="state",
    tools=(ToolInfo(name="/archive_email"),), state_sample={"emails": []},
)
TAXONOMY = Taxonomy(
    categories=[TaxonomyCategory(name="triage", description="sort mail", exercises=[], difficulties=[1, 5])],
    difficulty_rubric={"1": "one step"},
)


def _draft(slot: int, title: str = "") -> TaskDraft:
    return TaskDraft(
        slot=slot, title=title or f"task {slot}", objective="archive it",
        golden=[GoldenStep(tool="/archive_email", args={"email_id": "e1"})],
        checks=[Check(kind="record_exists", collection="emails", match={"id": "e1"}, expect={"folder": "archive"})],
    )


class _EchoWriterClient:
    """Answers each chunk with one draft per slot named in the prompt."""

    def __init__(self, skip: set[int] = frozenset(), extra: int | None = None) -> None:
        self.prompts: list[str] = []
        self._skip = skip
        self._extra = extra

    def extract(self, system, user, schema):
        self.prompts.append(user)
        slots = [int(line.split()[1]) for line in user.splitlines() if line.startswith("SLOT ")]
        drafts = [_draft(s) for s in slots if s not in self._skip]
        if self._extra is not None:
            drafts.append(_draft(self._extra))
        return TaskDraftBatch(tasks=drafts)


def _slots(n: int, difficulty: int = 1) -> list[Slot]:
    return [Slot(index=i, category="triage", difficulty=difficulty) for i in range(n)]


def test_the_writer_fills_slots_in_chunks_of_five():
    client = _EchoWriterClient()

    drafts = TaskWriter(client).write(PROFILE, TAXONOMY, _slots(23))

    assert len(client.prompts) == 5
    assert sorted(drafts) == list(range(23))


def test_a_slot_the_writer_skipped_is_missing_from_the_result():
    drafts = TaskWriter(_EchoWriterClient(skip={4})).write(PROFILE, TAXONOMY, _slots(6))

    assert 4 not in drafts
    assert len(drafts) == 5


def test_a_draft_for_a_slot_outside_the_chunk_is_dropped():
    drafts = TaskWriter(_EchoWriterClient(extra=99)).write(PROFILE, TAXONOMY, _slots(3))

    assert 99 not in drafts


def test_the_prompt_carries_feedback_titles_and_the_step_floor():
    client = _EchoWriterClient()

    TaskWriter(client).write(
        PROFILE, TAXONOMY, [Slot(index=7, category="triage", difficulty=5)],
        avoid_titles=["Archive the Q3 invoice"],
        feedback={7: "golden step 3 failed: no such email"},
    )

    prompt = client.prompts[0]
    assert "Archive the Q3 invoice" in prompt
    assert "golden step 3 failed" in prompt
    assert "30 or more" in prompt
    assert "aim for about 45" in prompt
    assert "/archive_email" in prompt


class _ScriptedReviewClient:
    """Answers through the schema the reviewer asks for, so an answer it would not accept fails here too."""

    def __init__(self, answers: list[dict]) -> None:
        self._answers = answers
        self.prompts: list[str] = []

    def extract(self, system, user, schema):
        self.prompts.append(user)
        self.schema = schema
        return schema.model_validate({"verdicts": self._answers})


# _draft's one check archives e1, which its objective "archive it" asks for.
TRACED = [{"check": 0, "requires": "e1 is in the archive folder", "asked_by": "archive it"}]


def _verdict(slot: int, fair: bool = True, traces: list[dict] = TRACED) -> dict:
    return {
        "slot": slot, "realistic": True, "realistic_reason": "real users do this",
        "fair": fair, "fair_reason": "ok" if fair else "the objective hides a required fact",
        "sensible": True, "sensible_reason": "coherent",
        "edge_case_ok": None, "edge_case_reason": "", "check_traces": traces,
    }


def test_the_reviewer_maps_verdicts_back_to_slots():
    client = _ScriptedReviewClient([_verdict(0), _verdict(1, fair=False)])

    verdicts = TaskReviewer(client).review(PROFILE, [(s, _draft(s.index)) for s in _slots(2)])

    assert verdicts[0].accepted
    assert not verdicts[1].accepted
    assert "hides a required fact" in verdicts[1].rejection_reason()


def test_a_task_the_reviewer_skipped_gets_no_verdict():
    client = _ScriptedReviewClient([_verdict(0)])

    verdicts = TaskReviewer(client).review(PROFILE, [(s, _draft(s.index)) for s in _slots(2)])

    assert 1 not in verdicts


def test_the_review_prompt_shows_the_golden_solution_and_reflections():
    draft = _draft(0).model_copy(update={
        "reflection_points": [ReflectionPoint(step=0, kind="verify", note="confirm the sender first")],
    })
    client = _ScriptedReviewClient([_verdict(0)])

    TaskReviewer(client).review(PROFILE, [(Slot(index=0, category="triage", difficulty=1), draft)])

    assert "/archive_email" in client.prompts[0]
    assert "confirm the sender first" in client.prompts[0]


def test_a_revision_shows_the_draft_its_problem_and_the_target():
    client = _EchoWriterClient()
    draft = _draft(3, title="Archive old invoices")

    TaskWriter(client).revise(
        PROFILE, TAXONOMY,
        [(Slot(index=3, category="triage", difficulty=5), draft, "difficulty 5 needs 30 or more steps, got 12")],
    )

    prompt = client.prompts[0]
    assert "Archive old invoices" in prompt
    assert "got 12" in prompt
    assert "aim for about 45" in prompt
    assert "SLOT 3" in prompt


def test_an_edge_case_slot_tells_the_writer_what_to_plant():
    client = _EchoWriterClient()

    TaskWriter(client).write(
        PROFILE, TAXONOMY,
        [Slot(index=0, category="triage", difficulty=1, edge_case="conflicting_state"),
         Slot(index=1, category="triage", difficulty=1)],
    )

    slot_lines = [line for line in client.prompts[0].splitlines() if line.startswith("SLOT ")]
    assert "edge case conflicting_state" in slot_lines[0]
    assert "edge case" not in slot_lines[1]
    assert "partly done" in client.prompts[0]


def test_a_plain_batch_prompt_carries_no_edge_case_descriptions():
    client = _EchoWriterClient()

    TaskWriter(client).write(PROFILE, TAXONOMY, _slots(2))

    assert "partly done" not in client.prompts[0]


def test_a_revision_keeps_the_slots_edge_case():
    client = _EchoWriterClient()

    TaskWriter(client).revise(
        PROFILE, TAXONOMY,
        [(Slot(index=3, category="triage", difficulty=5, edge_case="recovery"), _draft(3), "too short")],
    )

    assert "edge case recovery" in client.prompts[0]


def _edge_verdict(slot: int, edge_case_ok: bool | None) -> dict:
    return _verdict(slot) | {
        "edge_case_ok": edge_case_ok,
        "edge_case_reason": "the seed never plants the stale record" if edge_case_ok is False else "",
    }


def _edge_slot(index: int, edge_case: str | None = "conflicting_state") -> Slot:
    return Slot(index=index, category="triage", difficulty=1, edge_case=edge_case)


def test_a_failed_edge_case_verdict_rejects_the_task():
    client = _ScriptedReviewClient([_edge_verdict(0, False)])

    verdicts = TaskReviewer(client).review(PROFILE, [(_edge_slot(0), _draft(0))])

    assert not verdicts[0].accepted
    assert "never plants the stale record" in verdicts[0].rejection_reason()


def test_an_edge_case_task_without_an_edge_case_verdict_is_rejected():
    client = _ScriptedReviewClient([_edge_verdict(0, None)])

    verdicts = TaskReviewer(client).review(PROFILE, [(_edge_slot(0), _draft(0))])

    assert not verdicts[0].accepted
    assert "edge case" in verdicts[0].rejection_reason()


class _CallByCallReviewClient:
    """Gives each extract call its own answers, in order."""

    def __init__(self, *calls: list[dict]) -> None:
        self._calls = list(calls)
        self.prompts: list[str] = []

    def extract(self, system, user, schema):
        self.prompts.append(user)
        return schema.model_validate({"verdicts": self._calls[len(self.prompts) - 1]})


def test_a_blank_edge_case_answer_is_asked_again_for_that_task_only():
    client = _CallByCallReviewClient(
        [_edge_verdict(0, None), _edge_verdict(1, True)],
        [_edge_verdict(0, True)],
    )

    verdicts = TaskReviewer(client).review(PROFILE, [(_edge_slot(0), _draft(0)), (_edge_slot(1), _draft(1))])

    assert verdicts[0].accepted and verdicts[1].accepted
    assert len(client.prompts) == 2
    assert "slot 0" in client.prompts[1] and "slot 1" not in client.prompts[1]


def test_a_second_blank_edge_case_answer_rejects_the_task_as_unanswered():
    client = _CallByCallReviewClient([_edge_verdict(0, None)], [_edge_verdict(0, None)])

    verdicts = TaskReviewer(client).review(PROFILE, [(_edge_slot(0), _draft(0))])

    assert not verdicts[0].accepted
    assert "gave no verdict on edge case conflicting_state" in verdicts[0].rejection_reason()
    assert len(client.prompts) == 2


def test_a_retry_that_skips_the_task_keeps_it_rejected():
    client = _CallByCallReviewClient([_edge_verdict(0, None)], [])

    verdicts = TaskReviewer(client).review(PROFILE, [(_edge_slot(0), _draft(0))])

    assert not verdicts[0].accepted
    assert "gave no verdict on edge case" in verdicts[0].rejection_reason()


def test_the_retry_takes_a_no_answer_at_its_word():
    client = _CallByCallReviewClient([_edge_verdict(0, None)], [_edge_verdict(0, False)])

    verdicts = TaskReviewer(client).review(PROFILE, [(_edge_slot(0), _draft(0))])

    assert not verdicts[0].accepted
    assert "never plants the stale record" in verdicts[0].rejection_reason()


def test_answered_edge_cases_and_plain_tasks_are_not_asked_again():
    client = _CallByCallReviewClient([_edge_verdict(0, False), _edge_verdict(1, None)])

    TaskReviewer(client).review(PROFILE, [(_edge_slot(0), _draft(0)), (_edge_slot(1, None), _draft(1))])

    assert len(client.prompts) == 1


def test_a_confirmed_edge_case_is_accepted():
    client = _ScriptedReviewClient([_edge_verdict(0, True)])

    verdicts = TaskReviewer(client).review(PROFILE, [(_edge_slot(0), _draft(0))])

    assert verdicts[0].accepted


def test_an_edge_case_verdict_on_a_plain_task_is_ignored():
    client = _ScriptedReviewClient([_edge_verdict(0, False)])

    verdicts = TaskReviewer(client).review(PROFILE, [(_edge_slot(0, None), _draft(0))])

    assert verdicts[0].accepted


def test_the_review_prompt_names_the_edge_case_and_its_seed():
    draft = _draft(0).model_copy(update={"edge_case": "conflicting_state", "edge_case_seed": ["emails:e1"]})
    client = _ScriptedReviewClient([_edge_verdict(0, True)])

    TaskReviewer(client).review(PROFILE, [(_edge_slot(0), draft)])

    assert "edge case conflicting_state" in client.prompts[0]
    assert "emails:e1" in client.prompts[0]


# ---------------------------------------------------------------------------
# A check must test something the objective asks for. A task whose check
# wants more than the objective says fails an agent that did what it was told.
# ---------------------------------------------------------------------------

def _traced_review(traces: list[dict], draft: TaskDraft | None = None):
    client = _ScriptedReviewClient([_verdict(0, traces=traces)])
    return TaskReviewer(client).review(PROFILE, [(_slots(1)[0], draft or _draft(0))])[0]


def test_a_check_the_objective_never_asks_for_rejects_the_task():
    verdict = _traced_review([{"check": 0, "requires": "the pid file is deleted", "asked_by": ""}])

    assert not verdict.accepted
    assert "check 0" in verdict.rejection_reason()
    assert "the pid file is deleted" in verdict.rejection_reason()


def test_a_check_traced_to_words_the_objective_does_not_contain_rejects_the_task():
    verdict = _traced_review([{"check": 0, "requires": "e1 is archived", "asked_by": "delete the pid file"}])

    assert not verdict.accepted
    assert "e1 is archived" in verdict.rejection_reason()


def test_a_check_the_validator_never_traced_rejects_the_task():
    draft = _draft(0).model_copy(update={"checks": _draft(0).checks * 2})

    verdict = _traced_review(TRACED, draft)

    assert not verdict.accepted
    assert "check 1" in verdict.rejection_reason()


def test_a_task_whose_checks_all_trace_to_the_objective_is_accepted():
    verdict = _traced_review(TRACED)

    assert verdict.accepted
    assert verdict.unstated_requirements == []


def test_an_objective_quote_matches_regardless_of_case_spacing_and_quote_marks():
    verdict = _traced_review([{"check": 0, "requires": "e1 is archived", "asked_by": "'Archive   It.'"}])

    assert verdict.accepted


def test_a_quote_cut_with_an_ellipsis_matches_when_each_piece_is_in_the_objective_in_order():
    draft = _draft(0).model_copy(update={"objective": "Find the invoice from Acme, then archive it today."})

    kept = _traced_review([{"check": 0, "requires": "archived", "asked_by": "Find the invoice... archive it"}], draft)
    reordered = _traced_review([{"check": 0, "requires": "archived", "asked_by": "archive it… Find the invoice"}], draft)

    assert kept.accepted
    assert not reordered.accepted


def test_a_quote_of_only_an_ellipsis_grounds_nothing():
    verdict = _traced_review([{"check": 0, "requires": "e1 is archived", "asked_by": "..."}])

    assert not verdict.accepted


def test_a_stored_verdict_from_before_the_check_comparison_still_loads_as_accepted():
    stored = ReviewVerdict.model_validate(_verdict(0) | {"check_traces": None}).model_dump(
        exclude={"unstated_requirements"}
    )

    assert ReviewVerdict.model_validate(stored).accepted


def test_the_validator_must_trace_checks_and_answer_the_edge_case():
    client = _ScriptedReviewClient([_verdict(0)])

    TaskReviewer(client).review(PROFILE, [(s, _draft(s.index)) for s in _slots(1)])

    schema = client.schema.model_json_schema()
    answer = schema["$defs"][schema["properties"]["verdicts"]["items"]["$ref"].split("/")[-1]]
    assert {"check_traces", "edge_case_ok"} <= set(answer["required"])


def test_the_review_prompt_numbers_the_checks():
    draft = _draft(0).model_copy(update={"checks": _draft(0).checks * 2})
    client = _ScriptedReviewClient([_verdict(0)])

    TaskReviewer(client).review(PROFILE, [(_slots(1)[0], draft)])

    assert "CHECK 0:" in client.prompts[0]
    assert "CHECK 1:" in client.prompts[0]


def test_the_validator_and_the_writer_are_told_checks_follow_the_objective():
    from forge.taskfactory import review, writer

    assert "asked_by" in review._SYSTEM
    assert "objective asks for" in writer._COMMON
