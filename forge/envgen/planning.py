from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.context import EnvGenContext


class AgentTask(BaseModel):
    id: str
    agent_id: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    context_keys: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class GenerationPlan(BaseModel):
    user_request: str
    tasks: list[AgentTask]

    @model_validator(mode="after")
    def validate_graph(self) -> GenerationPlan:
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Generation plan contains duplicate task ids")
        known = set(task_ids)
        for task in self.tasks:
            missing = set(task.dependencies) - known
            if missing:
                raise ValueError(f"Task {task.id!r} has unknown dependencies: {sorted(missing)}")
            if task.id in task.dependencies:
                raise ValueError(f"Task {task.id!r} cannot depend on itself")
        visiting: set[str] = set()
        visited: set[str] = set()
        by_id = {task.id: task for task in self.tasks}

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("Generation plan contains a dependency cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in by_id[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in known:
            visit(task_id)
        return self


class PromptPlannerAgent:
    """Turns the extracted user prompt into an executable specialist todo list."""

    agent_id = "prompt_planner"

    def create_plan(
        self, ctx: EnvGenContext, agents: list[EnvGenAgent]
    ) -> GenerationPlan:
        producers: dict[str, str] = {}
        normalized_outputs: dict[str, list[str]] = {}
        for agent in agents:
            outputs = self._as_list(agent.produces)
            normalized_outputs[agent.agent_id] = outputs
            for artifact in outputs:
                if artifact in producers:
                    raise ValueError(f"Multiple agents produce artifact {artifact!r}")
                producers[artifact] = agent.agent_id

        tasks: list[AgentTask] = []
        for agent in agents:
            inputs = self._as_list(agent.depends_on)
            dependency_agents = list(dict.fromkeys(
                producers[name] for name in inputs if name in producers
            ))
            missing_inputs = [name for name in inputs if name not in producers]
            if missing_inputs:
                raise ValueError(
                    f"Agent {agent.agent_id!r} requires artifacts with no producer: "
                    f"{missing_inputs}"
                )
            tasks.append(AgentTask(
                id=agent.agent_id,
                agent_id=agent.agent_id,
                description=self._description(agent.agent_id, ctx),
                dependencies=dependency_agents,
                context_keys=inputs,
                outputs=normalized_outputs[agent.agent_id],
                acceptance_criteria=self._criteria(agent.agent_id, ctx),
            ))
        return GenerationPlan(user_request=ctx.description, tasks=tasks)

    @staticmethod
    def _as_list(value: list[str] | str) -> list[str]:
        return [value] if isinstance(value, str) else list(value)

    @staticmethod
    def _description(agent_id: str, ctx: EnvGenContext) -> str:
        descriptions = {
            "backend_builder": "Build the environment API, persistence, and container files.",
            "ui_builder": "Build the domain-specific user interface and client interactions.",
            "app_assembler": "Combine backend and UI outputs without changing either concern.",
            "telemetry": "Instrument state-changing API operations for episode telemetry.",
            "state_bridge": "Expose generated application state as an RL environment.",
            "policy": "Create policy constraints for the available actions.",
            "reward": "Create reward logic aligned with the requested RL behavior.",
            "reviewer": "Review requirement coverage and generated code quality.",
        }
        return descriptions.get(agent_id, f"Run the {agent_id} specialist for {ctx.env_name}.")

    @staticmethod
    def _criteria(agent_id: str, ctx: EnvGenContext) -> list[str]:
        if agent_id == "reviewer":
            return [
                "All required artifacts are present and non-empty.",
                "Generated Python parses successfully.",
                "The API, UI, actions, policy, reward, and RL bridge satisfy the user request.",
            ]
        if agent_id == "ui_builder":
            return [f"The UI represents the {ctx.compiler_input.domain} domain.",
                    "Every declared action is available through the interface."]
        return [f"Outputs are complete and usable for: {ctx.description}"]
