from __future__ import annotations

import asyncio

from forge.envgen.a2a import AgentMessage, MessageKind
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.planning import GenerationPlan
from forge.envgen.error_handling import GenerationErrorHandler


class TaskExecutor:
    """Executes a generation plan while enforcing task dependencies and scopes."""

    def __init__(self, error_handler: GenerationErrorHandler | None = None) -> None:
        self.error_handler = error_handler or GenerationErrorHandler()

    async def execute(
        self,
        plan: GenerationPlan,
        agents: list[EnvGenAgent],
        ctx: EnvGenContext,
        bus: ArtifactBus,
    ) -> None:
        agent_by_id = {agent.agent_id: agent for agent in agents}
        task_by_id = {task.id: task for task in plan.tasks}
        running: dict[str, asyncio.Task[None]] = {}

        async def run_task(task_id: str) -> None:
            task = task_by_id[task_id]
            await asyncio.gather(*(get_task(dep) for dep in task.dependencies))
            agent = agent_by_id.get(task.agent_id)
            if agent is None:
                raise ValueError(f"No specialist registered for {task.agent_id!r}")
            bus.protocol.send(AgentMessage(
                sender="orchestrator",
                recipient=task.agent_id,
                kind=MessageKind.TASK_ASSIGNED,
                task_id=task.id,
                payload={
                    "description": task.description,
                    "acceptance_criteria": task.acceptance_criteria,
                    "context_keys": task.context_keys,
                },
            ))
            channel = bus.scoped(
                agent_id=task.agent_id,
                task_id=task.id,
                readable=set(task.context_keys),
                writable=set(task.outputs),
            )
            try:
                await agent.run(ctx, channel)  # type: ignore[arg-type]
            except Exception as exc:
                handled = self.error_handler.capture(
                    task_id=task.id,
                    agent_id=task.agent_id,
                    error=exc,
                )
                bus.protocol.send(AgentMessage(
                    sender=task.agent_id,
                    recipient="orchestrator",
                    kind=MessageKind.TASK_FAILED,
                    task_id=task.id,
                    payload={"error": str(handled), "error_type": type(exc).__name__},
                ))
                raise handled from exc
            bus.protocol.send(AgentMessage(
                sender=task.agent_id,
                recipient="orchestrator",
                kind=(MessageKind.REVIEW_COMPLETED
                      if task.agent_id == "reviewer" else MessageKind.TASK_COMPLETED),
                task_id=task.id,
                payload={"outputs": task.outputs},
            ))

        def get_task(task_id: str) -> asyncio.Task[None]:
            if task_id not in running:
                running[task_id] = asyncio.create_task(run_task(task_id))
            return running[task_id]

        await asyncio.gather(*(get_task(task.id) for task in plan.tasks))
