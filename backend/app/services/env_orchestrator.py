from __future__ import annotations
import asyncio
from typing import Any, Callable, Awaitable
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.agents.app_generator import (
    AppAssemblyAgent,
    BackendBuilderAgent,
    UIBuilderAgent,
)
from forge.envgen.agents.reviewer import GenerationReview, GenerationReviewError, ReviewerAgent
from forge.envgen.agents.telemetry import TelemetryAgent
from forge.envgen.agents.state_bridge import StateBridgeAgent
from forge.envgen.agents.policy import PolicyAgent
from forge.envgen.agents.reward import RewardAgent
from forge.envgen.research import UserResearchAgent
from forge.extraction.schemas import CompilerInput
from forge.paths import confined_path, confined_relative_path, validate_path_segment
from forge.settings import generated_envs_root
from forge.envgen.executor import TaskExecutor
from forge.envgen.planning import PromptPlannerAgent


class EnvironmentOrchestrator:
    def __init__(
        self,
        agents: list[EnvGenAgent] | None = None,
        on_progress: Callable[[str, Any], Awaitable[None]] | None = None,
        on_log: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._agents = agents
        self._on_progress = on_progress
        self._on_log = on_log

    async def run(
        self,
        env_name: str,
        description: str,
        compiler_input: CompilerInput,
        policy_requirements: str = "",
        reward_requirements: str = "",
        reference_urls: list[str] | None = None,
        use_user_researcher: bool = False,
        source_product_name: str = "",
        source_product_url: str = "",
    ) -> None:
        ctx = EnvGenContext(
            env_name=env_name,
            description=description,
            compiler_input=compiler_input,
            policy_requirements=policy_requirements,
            reward_requirements=reward_requirements,
            reference_urls=reference_urls or [],
            source_product_name=source_product_name,
            source_product_url=source_product_url,
        )
        bus = ArtifactBus()
        if self._on_progress:
            bus.on_publish(self._on_progress)
        if self._on_log:
            bus.on_log(self._on_log)

        agents = self._agents or [
            *([UserResearchAgent()] if use_user_researcher else []),
            BackendBuilderAgent(),
            UIBuilderAgent(),
            AppAssemblyAgent(),
            TelemetryAgent(),
            StateBridgeAgent(),
            PolicyAgent(),
            RewardAgent(),
            ReviewerAgent(),
        ]

        # Explicitly supplied agents retain the open bus used by extensions and
        # older tests. The built-in pipeline uses planned tasks and scoped A2A
        # channels so each specialist sees only declared dependencies.
        if self._agents is not None:
            await asyncio.gather(*[agent.run(ctx, bus) for agent in agents])
        else:
            plan = PromptPlannerAgent().create_plan(ctx, agents)
            await bus.publish("generation_plan", plan)
            await TaskExecutor().execute(plan, agents, ctx, bus)
            review: GenerationReview | None = bus.get("review_report")
            if review is None:
                raise RuntimeError("Reviewer did not publish a review report")
            if not review.approved:
                raise GenerationReviewError(review)
        self._write_artifacts(env_name, bus)

    @staticmethod
    def _write_artifacts(env_name: str, bus: ArtifactBus) -> None:
        envs_root = generated_envs_root()
        validate_path_segment(env_name, label="environment name")
        pkg_dir = confined_path(envs_root, env_name)
        app_dir = pkg_dir / "app"
        app_dir.mkdir(parents=True, exist_ok=True)
        custom_dir = pkg_dir / "custom"
        custom_dir.mkdir(exist_ok=True)

        # Write all original files first so nothing is lost if the telemetry
        # agent only returns the files it touched.
        app_code: dict[str, str] = bus.get("app_code") or {}
        for path, content in app_code.items():
            dest = confined_relative_path(app_dir, path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        # Overlay with instrumented versions (these take precedence).
        instrumented: dict[str, str] = bus.get("instrumented_code") or {}
        for path, content in instrumented.items():
            dest = confined_relative_path(app_dir, path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        state_bridge: str = bus.get("state_bridge_code") or ""
        if state_bridge:
            (pkg_dir / "container_env.py").write_text(state_bridge)

        policy_dsl: str = bus.get("policy_dsl") or ""
        if policy_dsl:
            (custom_dir / "policies.yaml").write_text(policy_dsl)

        reward_fn: str = bus.get("reward_fn_code") or ""
        if reward_fn:
            (pkg_dir / "reward_fn.py").write_text(reward_fn)

        manifest = bus.get("state_schema_manifest")
        if manifest is not None:
            (pkg_dir / "state_schema.json").write_text(manifest.model_dump_json())
