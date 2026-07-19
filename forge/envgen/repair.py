from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from forge.envgen.a2a import AgentMessage, MessageKind
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.agents.reviewer import (
    GenerationReview,
    GenerationReviewError,
    ReviewIssue,
    ReviewSeverity,
)
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.config import envgen_config
from forge.envgen.context import EnvGenContext
from forge.envgen.executor import TaskExecutor
from forge.envgen.planning import GenerationPlan

_REVIEWERS = ("correctness_reviewer", "reviewer")
_CORRECTION_PREFIX = "correction:"


def correction_key(agent_id: str) -> str:
    """Bus artifact name carrying a specialist's correction envelope."""
    return f"{_CORRECTION_PREFIX}{agent_id}"

# app_code is a composite artifact assembled from the backend and UI builders.
# Findings against its files must route to the source specialist, not the
# pure-combine assembler that merely republishes them.
_APP_ASSEMBLER = "app_assembler"
_UI_FILE = "ui.html"
_INSTRUMENTED_PREFIX = "instrumented:"
_CODE_FILE_SUFFIXES = (
    ".py", ".html", ".txt", ".json", ".yaml", ".yml",
    ".md", ".cfg", ".ini", ".toml", ".sh",
)


def _looks_like_app_file(artifact: str) -> bool:
    return (
        artifact == "Dockerfile"
        or "/" in artifact
        or artifact.endswith(_CODE_FILE_SUFFIXES)
    )


class FindingRouter:
    """Maps a review finding to the specialist responsible for its artifact."""

    def __init__(self, agents: list[EnvGenAgent]) -> None:
        self._present: set[str] = {agent.agent_id for agent in agents}
        self._producers: dict[str, str] = {}
        for agent in agents:
            for artifact in agent.produces:
                self._producers[artifact] = agent.agent_id

    def route(self, issue: ReviewIssue) -> str | None:
        artifact = issue.artifact
        if not artifact:
            return None
        if artifact.startswith(_INSTRUMENTED_PREFIX):
            return self._present_or_none("telemetry")
        if _looks_like_app_file(artifact):
            if artifact.lower() == _UI_FILE:
                return self._present_or_none("ui_builder")
            return self._present_or_none("backend_builder")
        producer = self._producers.get(artifact)
        if producer is None or producer == _APP_ASSEMBLER:
            return None
        return producer

    def _present_or_none(self, agent_id: str) -> str | None:
        return agent_id if agent_id in self._present else None


class RepairPlanner:
    """Builds a self-contained re-run sub-plan for a set of targeted specialists."""

    def __init__(self, base_plan: GenerationPlan) -> None:
        self._base = base_plan
        self._by_id = {task.id: task for task in base_plan.tasks}
        # Reverse edges: producer id -> direct consumer ids.
        self._consumers: dict[str, set[str]] = {task.id: set() for task in base_plan.tasks}
        for task in base_plan.tasks:
            for dep in task.dependencies:
                self._consumers[dep].add(task.id)

    def subplan(self, target_agent_ids: set[str]) -> GenerationPlan:
        rerun = set(target_agent_ids)
        for target in target_agent_ids:
            rerun |= self._downstream(target)
        rerun |= {r for r in _REVIEWERS if r in self._by_id}

        tasks = [
            task.model_copy(update={
                "dependencies": [d for d in task.dependencies if d in rerun],
                # Targets read their correction envelope; downstream tasks do not.
                "context_keys": (
                    [*task.context_keys, correction_key(task.id)]
                    if task.id in target_agent_ids
                    else list(task.context_keys)
                ),
            })
            for task in self._base.tasks
            if task.id in rerun
        ]
        return GenerationPlan(user_request=self._base.user_request, tasks=tasks)

    def _downstream(self, start: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self._consumers.get(start, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self._consumers.get(node, ()))
        return seen


class CorrectionTask(BaseModel):
    """A single rejected finding routed to the specialist that owns its artifact."""

    finding: ReviewIssue
    target_agent_id: str
    artifact: str | None = None
    acceptance_criteria: list[str] = []
    source_report: str
    round: int


class UnrepairableFinding(Exception):
    """Raised when a finding maps to no specialist and cannot be repaired."""

    def __init__(self, finding: ReviewIssue) -> None:
        super().__init__(
            f"No specialist can repair finding on {finding.artifact!r}: {finding.message}"
        )
        self.finding = finding


def correction_tasks_for(
    findings: list[tuple[ReviewIssue, str]],
    router: FindingRouter,
    plan: GenerationPlan,
    round_number: int,
) -> list[CorrectionTask]:
    """Convert routed findings into typed correction tasks.

    Raises :class:`UnrepairableFinding` for any finding with no owning specialist.
    """
    by_id = {task.id: task for task in plan.tasks}
    tasks: list[CorrectionTask] = []
    for issue, source in findings:
        target = router.route(issue)
        if target is None:
            raise UnrepairableFinding(issue)
        criteria = list(by_id[target].acceptance_criteria) if target in by_id else []
        tasks.append(CorrectionTask(
            finding=issue,
            target_agent_id=target,
            artifact=issue.artifact,
            acceptance_criteria=criteria,
            source_report=source,
            round=round_number,
        ))
    return tasks


def _fingerprint(issue: ReviewIssue) -> tuple[str, str | None, str]:
    return (issue.category, issue.artifact, issue.message)


class RepairLoop:
    """Bounded, circuit-broken repair loop over reviewer/correctness findings.

    Runs after the initial pipeline pass. Each round converts ERROR findings into
    correction tasks, routes them to their specialists, re-runs the affected
    sub-graph plus the reviewers, and re-evaluates. Only ERROR findings drive
    repair; WARNINGs never spend an LLM round. The loop stops — raising
    :class:`GenerationReviewError` — on the first of: retry bound reached, a round
    that makes no progress (identical findings, breaker opens), or an unrepairable
    finding.
    """

    def __init__(self, *, max_repair_rounds: int | None = None) -> None:
        self._max = (
            max_repair_rounds
            if max_repair_rounds is not None
            else envgen_config().max_repair_rounds
        )

    async def run(
        self,
        plan: GenerationPlan,
        agents: list[EnvGenAgent],
        ctx: EnvGenContext,
        bus: ArtifactBus,
        executor: TaskExecutor,
    ) -> None:
        router = FindingRouter(agents)
        planner = RepairPlanner(plan)
        by_id = {task.id: task for task in plan.tasks}
        previous: Counter | None = None

        for round_number in range(self._max + 1):
            review: GenerationReview | None = bus.get("review_report")
            if review is None:
                raise RuntimeError("Reviewer did not publish a review report")
            correctness: GenerationReview | None = bus.get("correctness_report")

            findings = self._error_findings(review, correctness)
            if not findings:
                return  # every gate approved

            failing = review if not review.approved else correctness

            if round_number == self._max:
                self._exhausted(bus, "retry_bound", findings)
                raise GenerationReviewError(failing)

            fingerprints = Counter(_fingerprint(issue) for issue, _ in findings)
            if previous is not None and fingerprints == previous:
                self._exhausted(bus, "no_progress", findings)
                raise GenerationReviewError(failing)

            try:
                tasks = correction_tasks_for(findings, router, plan, round_number + 1)
            except UnrepairableFinding:
                self._exhausted(bus, "unrepairable", findings)
                raise GenerationReviewError(failing)

            self._announce_rejection(bus, findings, round_number + 1)
            targets = await self._publish_corrections(bus, tasks, by_id)

            subplan = planner.subplan(set(targets))
            bus.invalidate(self._outputs(subplan))
            await executor.execute(subplan, agents, ctx, bus)

            self._announce_completion(bus, targets, round_number + 1)
            bus.invalidate(correction_key(target) for target in targets)
            previous = fingerprints

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _error_findings(
        review: GenerationReview, correctness: GenerationReview | None
    ) -> list[tuple[ReviewIssue, str]]:
        findings: list[tuple[ReviewIssue, str]] = [
            (issue, "review")
            for issue in review.issues
            if issue.severity == ReviewSeverity.ERROR
        ]
        if correctness is not None:
            findings.extend(
                (issue, "correctness")
                for issue in correctness.issues
                if issue.severity == ReviewSeverity.ERROR
            )
        return findings

    async def _publish_corrections(
        self, bus: ArtifactBus, tasks: list[CorrectionTask], by_id: dict
    ) -> set[str]:
        grouped: dict[str, list[CorrectionTask]] = {}
        for task in tasks:
            grouped.setdefault(task.target_agent_id, []).append(task)

        for target, group in grouped.items():
            criteria = group[0].acceptance_criteria
            await bus.publish(correction_key(target), {
                "findings": [task.finding.model_dump() for task in group],
                "acceptance_criteria": criteria,
                "prior_output": self._prior_output(bus, by_id.get(target)),
            })
            for task in group:
                bus.protocol.send(AgentMessage(
                    sender="orchestrator",
                    recipient=target,
                    kind=MessageKind.CORRECTION_ASSIGNED,
                    task_id=target,
                    payload={
                        "finding": task.finding.model_dump(),
                        "acceptance_criteria": criteria,
                        "round": task.round,
                    },
                ))
        return set(grouped)

    @staticmethod
    def _prior_output(bus: ArtifactBus, task) -> object:
        if task is None or not task.outputs:
            return None
        if len(task.outputs) == 1:
            return bus.get(task.outputs[0])
        return {name: bus.get(name) for name in task.outputs}

    @staticmethod
    def _outputs(subplan: GenerationPlan) -> list[str]:
        names: list[str] = []
        for task in subplan.tasks:
            names.extend(task.outputs)
        return names

    @staticmethod
    def _announce_rejection(
        bus: ArtifactBus, findings: list[tuple[ReviewIssue, str]], round_number: int
    ) -> None:
        bus.protocol.send(AgentMessage(
            sender="reviewer",
            recipient="orchestrator",
            kind=MessageKind.REVIEW_REJECTED,
            task_id="reviewer",
            payload={
                "round": round_number,
                "findings": [issue.model_dump() for issue, _ in findings],
            },
        ))

    @staticmethod
    def _announce_completion(
        bus: ArtifactBus, targets: set[str], round_number: int
    ) -> None:
        for target in targets:
            bus.protocol.send(AgentMessage(
                sender=target,
                recipient="orchestrator",
                kind=MessageKind.CORRECTION_COMPLETED,
                task_id=target,
                payload={"round": round_number},
            ))

    @staticmethod
    def _exhausted(
        bus: ArtifactBus, reason: str, findings: list[tuple[ReviewIssue, str]]
    ) -> None:
        bus.protocol.send(AgentMessage(
            sender="orchestrator",
            recipient="*",
            kind=MessageKind.REPAIR_EXHAUSTED,
            task_id="orchestrator",
            payload={
                "reason": reason,
                "findings": [issue.model_dump() for issue, _ in findings],
            },
        ))
