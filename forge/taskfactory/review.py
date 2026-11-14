"""Step 3c: a model from another family judges each task realistic, fair, sensible."""
from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from forge.extraction.llm_client import LLMClient
from forge.taskfactory.profile import EnvironmentProfile
from forge.taskfactory.schemas import ReviewVerdict, TaskDraft
from forge.taskfactory.slots import Slot

CHUNK_SIZE = 10

_SYSTEM = """\
You review synthetic tasks written for a reinforcement-learning environment by
a different model. Every task has already been run: its golden solution passed
its checks on repeated runs. Judge what running cannot.

For each task, answer three questions with a verdict and a one-sentence reason:
  realistic: would a real user of this product ask for this, with this data?
  fair: does the objective give the agent every fact it cannot find in the
    environment, and do the checks reward solving the task rather than one
    particular way of solving it?
  sensible: is the task coherent, with an objective, seed, golden solution
    and checks that agree with each other? For difficulty 4 and 5, does each
    reflection point force the agent to stop and rethink, or could it skip it?

Then trace every check, in check_traces, one entry per CHECK number:
  requires: what the check needs to be true at the end, in plain words.
  asked_by: the objective's words, copied verbatim, that ask for the result
    the check tests. Leave it empty only when the objective never asks for
    that result.
A check that confirms the agent got a requested result right is asked for,
even when the objective does not state the answer. Quote the words that
request the result. Examples: the right count or values in a file the
objective asks for, a requested file being non-empty, a line in the section
the objective names, an old value gone after the update the objective asks
for, a value in the place and format the objective specifies.
A check is unstated when it needs an action or detail the objective never
requests, and so fails an agent that did exactly what it was told. Examples:
deleting, moving or keeping a file the objective never asks to act on, an
exact phrase or wording in a report whose format the objective leaves open, a
side effect nobody asked for. Words that only mention a file do not ask to
act on it: "check whether X exists" does not ask to delete X.

A task whose header names an edge case also gets edge_case_ok and
edge_case_reason: does its seed truly set up that condition (edge_case_seed
lists where), and does the golden solution handle it? Set edge_case_ok to
null for tasks without an edge case.

Be strict. Return one verdict per task with `slot` set to its slot number.
"""


class _CheckTrace(BaseModel):
    check: int
    requires: str
    asked_by: str


class _Answer(BaseModel):
    """A verdict as the validator must give it. Every field is required, because models skip optional ones."""

    slot: int
    realistic: bool
    realistic_reason: str
    fair: bool
    fair_reason: str
    sensible: bool
    sensible_reason: str
    edge_case_ok: bool | None
    edge_case_reason: str
    check_traces: list[_CheckTrace]


class _AnswerBatch(BaseModel):
    verdicts: list[_Answer]


class TaskReviewer:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def review(
        self, profile: EnvironmentProfile, tasks: Sequence[tuple[Slot, TaskDraft]]
    ) -> dict[int, ReviewVerdict]:
        """Return the verdicts the reviewer gave, keyed by slot index.

        Models sometimes leave the edge-case answer blank. Those tasks are asked
        once more, and a second blank rejects the task as unanswered.
        """
        verdicts = self._verdicts(profile, tasks)
        blank = [
            (slot, draft) for slot, draft in tasks
            if slot.edge_case and slot.index in verdicts and verdicts[slot.index].edge_case_ok is None
        ]
        if blank:
            retried = self._verdicts(profile, blank)
            for slot, _ in blank:
                if slot.index in retried:
                    verdicts[slot.index] = verdicts[slot.index].model_copy(update={
                        "edge_case_ok": retried[slot.index].edge_case_ok,
                        "edge_case_reason": retried[slot.index].edge_case_reason,
                    })
        slots = {slot.index: slot for slot, _ in tasks}
        return {index: _settle_edge_case(verdict, slots[index]) for index, verdict in verdicts.items()}

    def _verdicts(
        self, profile: EnvironmentProfile, tasks: Sequence[tuple[Slot, TaskDraft]]
    ) -> dict[int, ReviewVerdict]:
        verdicts: dict[int, ReviewVerdict] = {}
        for start in range(0, len(tasks), CHUNK_SIZE):
            chunk = tasks[start:start + CHUNK_SIZE]
            user = "\n\n".join([profile.prompt_view(), *[_task_view(slot, draft) for slot, draft in chunk]])
            batch = self._client.extract(system=_SYSTEM, user=user, schema=_AnswerBatch)
            wanted = {slot.index: (slot, draft) for slot, draft in chunk}
            for answer in batch.verdicts:
                if answer.slot in wanted and answer.slot not in verdicts:
                    _, draft = wanted[answer.slot]
                    verdicts[answer.slot] = ReviewVerdict(
                        **answer.model_dump(exclude={"check_traces"}),
                        unstated_requirements=_unstated(answer.check_traces, draft),
                    )
        return verdicts


def _settle_edge_case(verdict: ReviewVerdict, slot: Slot) -> ReviewVerdict:
    """An edge-case task needs an explicit yes. A plain task's edge-case answer means nothing."""
    if slot.edge_case is None:
        return verdict.model_copy(update={"edge_case_ok": None, "edge_case_reason": ""})
    if verdict.edge_case_ok is None:
        return verdict.model_copy(update={
            "edge_case_ok": False,
            "edge_case_reason": f"the validator gave no verdict on edge case {slot.edge_case}",
        })
    return verdict


def _unstated(traces: Sequence[_CheckTrace], draft: TaskDraft) -> list[str]:
    """Checks the objective never asks for, judged by code from the validator's traces.

    A trace counts only when its quote is really in the objective, so the
    validator cannot ground a check in words it made up.
    """
    objective = _words(draft.objective)
    first = {}
    for trace in traces:
        first.setdefault(trace.check, trace)
    unstated = []
    for number in range(len(draft.checks)):
        trace = first.get(number)
        if trace is None:
            unstated.append(f"check {number}: the validator never traced it to the objective")
        elif not _quoted(trace.asked_by, objective):
            unstated.append(f"check {number}: {trace.requires}")
    return unstated


def _quoted(quote: str, objective: str) -> bool:
    """True when the quote is in the objective. A quote cut with "..." needs each piece there, in order."""
    pieces = [_words(piece) for piece in re.split(r"\.\.\.|…", quote)]
    pieces = [piece for piece in pieces if piece]
    position = 0
    for piece in pieces:
        found = objective.find(piece, position)
        if found < 0:
            return False
        position = found + len(piece)
    return bool(pieces)


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _task_view(slot: Slot, draft: TaskDraft) -> str:
    body = draft.model_dump_json(exclude={"slot", "checks"}, indent=1)
    checks = "\n".join(
        f"CHECK {number}: {check.model_dump_json(exclude_none=True)}" for number, check in enumerate(draft.checks)
    )
    edge = f" | edge case {slot.edge_case}" if slot.edge_case else ""
    return f"TASK slot {slot.index} | category {slot.category} | difficulty {slot.difficulty}{edge}\n{body}\n{checks}"
