"""Step 3c: a model from another family judges each task realistic, fair, sensible."""
from __future__ import annotations

from collections.abc import Sequence

from forge.extraction.llm_client import LLMClient
from forge.taskfactory.profile import EnvironmentProfile
from forge.taskfactory.schemas import ReviewVerdict, ReviewVerdictBatch, TaskDraft
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

Be strict. Return one verdict per task with `slot` set to its slot number.
"""


class TaskReviewer:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def review(
        self, profile: EnvironmentProfile, tasks: Sequence[tuple[Slot, TaskDraft]]
    ) -> dict[int, ReviewVerdict]:
        """Return the verdicts the reviewer gave, keyed by slot index."""
        verdicts: dict[int, ReviewVerdict] = {}
        for start in range(0, len(tasks), CHUNK_SIZE):
            chunk = tasks[start:start + CHUNK_SIZE]
            user = "\n\n".join([profile.prompt_view(), *[_task_view(slot, draft) for slot, draft in chunk]])
            batch: ReviewVerdictBatch = self._client.extract(system=_SYSTEM, user=user, schema=ReviewVerdictBatch)
            wanted = {slot.index for slot, _ in chunk}
            for verdict in batch.verdicts:
                if verdict.slot in wanted and verdict.slot not in verdicts:
                    verdicts[verdict.slot] = verdict
        return verdicts


def _task_view(slot: Slot, draft: TaskDraft) -> str:
    body = draft.model_dump_json(exclude={"slot"}, indent=1)
    return f"TASK slot {slot.index} | category {slot.category} | difficulty {slot.difficulty}\n{body}"
