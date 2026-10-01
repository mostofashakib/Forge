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
    ReviewVerdictBatch,
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
    def __init__(self, verdicts: list[ReviewVerdict]) -> None:
        self._verdicts = verdicts
        self.prompts: list[str] = []

    def extract(self, system, user, schema):
        self.prompts.append(user)
        return ReviewVerdictBatch(verdicts=self._verdicts)


def _verdict(slot: int, fair: bool = True) -> ReviewVerdict:
    return ReviewVerdict(
        slot=slot, realistic=True, realistic_reason="real users do this",
        fair=fair, fair_reason="ok" if fair else "the objective hides a required fact",
        sensible=True, sensible_reason="coherent",
    )


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
