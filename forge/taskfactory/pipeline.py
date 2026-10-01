"""Run the four steps for one batch: taxonomy, writer, validation, top-up.

The registry step is the caller's: the pipeline returns a finished result
and never touches storage, so a failure midway leaves nothing half-saved.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Protocol

from forge.taskfactory.contamination import TaskContaminationVerifier
from forge.taskfactory.pass_k import PassKResult, run_pass_k
from forge.taskfactory.profile import EnvironmentProfile
from forge.taskfactory.runner import TaskRunner
from forge.taskfactory.schemas import (
    PreferencePair,
    ReviewVerdict,
    SFTItem,
    SyntheticTask,
    TaskDraft,
    TaskRejection,
    Taxonomy,
)
from forge.taskfactory.slots import Slot, plan_slots
from forge.taskfactory.static_check import length_problems, static_problems

MAX_ROUNDS = 3
MIN_COUNT, MAX_COUNT = 1, 20_000
# Slots written, run, and reviewed together. The environment lock is held
# for one chunk's golden runs at a time.
PROCESS_CHUNK = 10
# Titles from the same categories the writer sees, to avoid repeats without
# a prompt that grows with the batch.
TITLE_CONTEXT = 50
MIN_K, MAX_K = 1, 10

Progress = Callable[[dict], None]
PassK = Callable[[TaskRunner, TaskDraft, int], PassKResult]


class _TaxonomyBuilder(Protocol):
    def build(self, profile: EnvironmentProfile, count: int) -> Taxonomy: ...


class _Writer(Protocol):
    def write(self, profile, taxonomy, slots, *, avoid_titles=(), feedback=None) -> dict[int, TaskDraft]: ...

    def revise(self, profile, taxonomy, items) -> dict[int, TaskDraft]: ...


class _Reviewer(Protocol):
    def review(self, profile, tasks) -> dict[int, ReviewVerdict]: ...


@dataclass
class PipelineResult:
    requested: int
    taxonomy: Taxonomy
    tasks: list[SyntheticTask] = field(default_factory=list)
    rejections: list[TaskRejection] = field(default_factory=list)
    data_type: str = "rl_tasks"
    preference_pairs: list[PreferencePair] = field(default_factory=list)
    sft_items: list[SFTItem] = field(default_factory=list)

    @property
    def shortfall(self) -> int:
        return self.requested - len(self.tasks)

    @property
    def status(self) -> str:
        return "complete" if self.shortfall == 0 else "short"

    def to_preference_pairs(self) -> list[PreferencePair]:
        """Convert accepted tasks (preferred) and rejections (dispreferred) into preference pairs.

        Reuses the natural output of the existing generation and filtering pipeline
        without introducing a separate generation process.
        """
        pairs: list[PreferencePair] = []
        rejected_drafts = [r for r in self.rejections if r.draft is not None]
        for idx, task in enumerate(self.tasks):
            matching_rejection = next(
                (r for r in rejected_drafts if r.category == task.category),
                rejected_drafts[idx % len(rejected_drafts)] if rejected_drafts else None,
            )
            dispreferred_dict = (
                matching_rejection.draft.model_dump()
                if matching_rejection and matching_rejection.draft
                else {"objective": f"[Failed Draft] {task.objective}", "steps": []}
            )
            reason = (
                matching_rejection.reason
                if matching_rejection
                else "failed filtering verification"
            )
            pairs.append(
                PreferencePair(
                    id=f"pref_{task.id}",
                    task_prompt=task.objective,
                    preferred=task.model_dump(),
                    dispreferred=dispreferred_dict,
                    rejection_reason=reason,
                    category=task.category,
                    difficulty=task.difficulty,
                )
            )
        return pairs

    def to_sft_items(self) -> list[SFTItem]:
        """Convert accepted tasks into SFT imitation learning examples."""
        return [
            SFTItem(
                id=f"sft_{task.id}",
                task_prompt=task.objective,
                golden_solution=[step.model_dump() for step in task.golden],
                golden_solution_patch=[step.model_dump() for step in task.golden],
                category=task.category,
                difficulty=task.difficulty,
            )
            for task in self.tasks
        ]


class TaskFactoryPipeline:
    def __init__(
        self,
        *,
        profile: EnvironmentProfile,
        taxonomy_builder: _TaxonomyBuilder,
        writer: _Writer,
        reviewer: _Reviewer,
        runner: TaskRunner,
        pass_k: PassK = run_pass_k,
        exclusive: Callable[[], AbstractContextManager[None]] | None = None,
        progress: Progress | None = None,
        contamination_verifier: TaskContaminationVerifier | None = None,
    ) -> None:
        self._profile = profile
        self._taxonomy_builder = taxonomy_builder
        self._writer = writer
        self._reviewer = reviewer
        self._runner = runner
        self._pass_k = pass_k
        self._exclusive = exclusive or nullcontext
        self._progress = progress or (lambda _event: None)
        self._contamination_verifier = contamination_verifier or TaskContaminationVerifier()

    def run(self, count: int, k: int, data_type: str = "rl_tasks") -> PipelineResult:
        if not MIN_COUNT <= count <= MAX_COUNT:
            raise ValueError(f"count must be {MIN_COUNT} to {MAX_COUNT}, got {count}")
        if not MIN_K <= k <= MAX_K:
            raise ValueError(f"k must be {MIN_K} to {MAX_K}, got {k}")
        self._emit("taxonomy", f"Building a taxonomy for {count} tasks")
        taxonomy = self._taxonomy_builder.build(self._profile, count)
        self._emit("taxonomy", f"Taxonomy has {len(taxonomy.categories)} categories")
        result = PipelineResult(requested=count, taxonomy=taxonomy, data_type=data_type)
        pending = plan_slots(taxonomy, count)
        feedback: dict[int, str] = {}
        for round_number in range(1, MAX_ROUNDS + 1):
            if not pending:
                break
            pending, feedback = self._round(round_number, pending, feedback, k, result)
        result.tasks.sort(key=lambda task: task.id)
        if data_type in ("preference", "preference_pairs"):
            result.preference_pairs = result.to_preference_pairs()
        elif data_type in ("sft", "sft_data"):
            result.sft_items = result.to_sft_items()
        return result

    def _round(
        self, round_number: int, slots: list[Slot], feedback: dict[int, str], k: int, result: PipelineResult
    ) -> tuple[list[Slot], dict[int, str]]:
        """Run one round in chunks, so results land steadily and the lock is held briefly."""
        self._emit("writing", f"Round {round_number}: {len(slots)} slots to fill", round_number)
        rejected: dict[int, str] = {}
        for start in range(0, len(slots), PROCESS_CHUNK):
            chunk = slots[start:start + PROCESS_CHUNK]
            self._chunk(round_number, chunk, feedback, k, result, rejected)
            self._emit(
                "review", f"Round {round_number}: {len(result.tasks)} of {result.requested} tasks accepted",
                round_number, accepted=len(result.tasks),
            )
        return [slot for slot in slots if slot.index in rejected], rejected

    def _chunk(
        self, round_number: int, slots: list[Slot], feedback: dict[int, str], k: int,
        result: PipelineResult, rejected: dict[int, str],
    ) -> None:
        def reject(slot: Slot, stage: str, reason: str, draft: TaskDraft | None) -> None:
            rejected[slot.index] = reason
            result.rejections.append(TaskRejection(
                slot=slot.index, category=slot.category, difficulty=slot.difficulty,
                round=round_number, stage=stage, reason=reason, draft=draft,
            ))
            self._emit(
                stage, f"Round {round_number}: slot {slot.index + 1} rejected: {reason[:300]}", round_number,
                rejected=True,
            )

        self._emit("writing", f"Round {round_number}: writing {len(slots)} tasks", round_number)
        drafts = self._writer.write(
            self._profile, result.taxonomy, slots,
            avoid_titles=_recent_titles(result.tasks, {slot.category for slot in slots}),
            feedback={slot.index: feedback[slot.index] for slot in slots if slot.index in feedback},
        )
        taken = {_title_key(task.title) for task in result.tasks}
        survivors: list[tuple[Slot, TaskDraft]] = []

        def check(slot: Slot, draft: TaskDraft) -> list[str]:
            problems = static_problems(draft, slot, self._profile)
            if _title_key(draft.title) in taken:
                problems.append(f"duplicate of an accepted task titled {draft.title!r}")
            return problems

        def keep(slot: Slot, draft: TaskDraft) -> None:
            survivors.append((slot, draft))
            taken.add(_title_key(draft.title))

        off_length: list[tuple[Slot, TaskDraft, str]] = []
        for slot in slots:
            draft = drafts.get(slot.index)
            if draft is None:
                reject(slot, "writer", "the writer returned no task for this slot", None)
                continue
            problems = check(slot, draft)
            if not problems:
                keep(slot, draft)
            elif problems == length_problems(draft, slot):
                # Only the length is off: rework it now instead of losing a round.
                off_length.append((slot, draft, "; ".join(problems)))
            else:
                reject(slot, "static", "; ".join(problems), draft)
        if off_length:
            self._emit("static", f"Round {round_number}: reworking {len(off_length)} tasks of the wrong length", round_number)
            revised = self._writer.revise(self._profile, result.taxonomy, off_length)
            for slot, draft, problem in off_length:
                rework = revised.get(slot.index)
                problems = check(slot, rework) if rework is not None else [problem]
                if problems:
                    reject(slot, "static", "; ".join(problems), rework or draft)
                else:
                    keep(slot, rework)
        self._emit("static", f"Round {round_number}: {len(survivors)} of {len(slots)} passed the static check", round_number)

        fingerprints: dict[int, str] = {}
        if survivors:
            self._emit("pass_k", f"Round {round_number}: running {len(survivors)} golden solutions {k} times each", round_number)
            with self._exclusive(), self._runner.batch():
                for slot, draft in survivors:
                    outcome = self._pass_k(self._runner, draft, k)
                    if outcome.passed:
                        fingerprints[slot.index] = outcome.fingerprint
                    else:
                        reject(slot, "pass_k", outcome.reason, draft)
        executable = [(slot, draft) for slot, draft in survivors if slot.index in fingerprints]
        if not executable:
            return
        self._emit("review", f"Round {round_number}: reviewing {len(executable)} tasks", round_number)
        verdicts = self._reviewer.review(self._profile, executable)
        accepted_candidates: list[tuple[Slot, TaskDraft, ReviewVerdict]] = []
        for slot, draft in executable:
            verdict = verdicts.get(slot.index)
            if verdict is None:
                reject(slot, "review", "the validator returned no verdict for this task", draft)
            elif not verdict.accepted:
                reject(slot, "review", verdict.rejection_reason(), draft)
            else:
                accepted_candidates.append((slot, draft, verdict))

        if accepted_candidates:
            self._emit("contamination", f"Round {round_number}: checking {len(accepted_candidates)} tasks for contamination", round_number)
            for slot, draft, verdict in accepted_candidates:
                report = self._contamination_verifier.verify_draft(draft) if self._contamination_verifier else None
                if report and report.is_contaminated:
                    reject(slot, "contamination", "; ".join(report.reasons), draft)
                else:
                    result.tasks.append(
                        _accepted(slot, draft, verdict, round_number, fingerprints[slot.index], contamination_report=report)
                    )

    def _emit(self, stage: str, log: str, round_number: int | None = None, **extra) -> None:
        event = {"stage": stage, "log": log, **extra}
        if round_number is not None:
            event["round"] = round_number
        self._progress(event)


def _title_key(title: str) -> str:
    return " ".join(title.lower().split())


def _recent_titles(tasks: list[SyntheticTask], categories: set[str]) -> list[str]:
    return [task.title for task in tasks if task.category in categories][-TITLE_CONTEXT:]


def _accepted(
    slot: Slot,
    draft: TaskDraft,
    verdict: ReviewVerdict,
    round_number: int,
    fingerprint: str,
    contamination_report: Any | None = None,
) -> SyntheticTask:
    return SyntheticTask(
        id=f"t-{slot.index + 1:05d}",
        category=slot.category,
        difficulty=slot.difficulty,
        title=draft.title,
        objective=draft.objective,
        seed=draft.seed,
        golden=draft.golden,
        checks=draft.checks,
        reflection_points=draft.reflection_points,
        step_budget=2 * len(draft.golden),
        round=round_number,
        review=verdict,
        fingerprint=fingerprint,
        contamination_report=contamination_report,
    )
