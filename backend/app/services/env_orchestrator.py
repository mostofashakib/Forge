from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Awaitable
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.agents.app_generator import AppGeneratorAgent
from forge.envgen.agents.telemetry import TelemetryAgent
from forge.envgen.agents.state_bridge import StateBridgeAgent
from forge.envgen.agents.policy import PolicyAgent
from forge.envgen.agents.reward import RewardAgent
from forge.extraction.schemas import CompilerInput


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
    ) -> None:
        ctx = EnvGenContext(
            env_name=env_name,
            description=description,
            compiler_input=compiler_input,
            policy_requirements=policy_requirements,
            reward_requirements=reward_requirements,
        )
        bus = ArtifactBus()
        if self._on_progress:
            bus.on_publish(self._on_progress)
        if self._on_log:
            bus.on_log(self._on_log)

        agents = self._agents or [
            AppGeneratorAgent(),
            TelemetryAgent(),
            StateBridgeAgent(),
            PolicyAgent(),
            RewardAgent(),
        ]
        await asyncio.gather(*[agent.run(ctx, bus) for agent in agents])
        self._write_artifacts(env_name, bus)

    @staticmethod
    def _write_artifacts(env_name: str, bus: ArtifactBus) -> None:
        envs_root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
        pkg_dir = envs_root / env_name
        app_dir = pkg_dir / "app"
        app_dir.mkdir(parents=True, exist_ok=True)
        custom_dir = pkg_dir / "custom"
        custom_dir.mkdir(exist_ok=True)

        instrumented: dict[str, str] = bus.get("instrumented_code") or {}
        for path, content in instrumented.items():
            dest = app_dir / path
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
