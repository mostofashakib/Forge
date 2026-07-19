import asyncio
import pytest
from pathlib import Path
from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.extraction.schemas import CompilerInput
from backend.app.services.env_orchestrator import EnvironmentOrchestrator


class _MockAgent(EnvGenAgent):
    def __init__(self, produces_name: str, value, depends: list[str] | None = None):
        self.produces = produces_name
        self.depends_on = depends or []
        self._value = value

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        for dep in self.depends_on:
            await bus.wait_for(dep)
        await bus.publish(self.produces, self._value)


def _compiler_input():
    return CompilerInput(
        project_name="test_env", domain="test", entities=[], actions=[], tasks=[]
    )


@pytest.mark.asyncio
async def test_orchestrator_runs_all_agents_concurrently(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path))
    agents = [
        _MockAgent("app_code", {"main.py": "# app"}),
        _MockAgent("instrumented_code", {"main.py": "# instrumented"}, depends=["app_code"]),
        _MockAgent("state_bridge_code", "class ContainerForgeEnv: pass", depends=["instrumented_code"]),
        _MockAgent("policy_dsl", "policies: []"),
        _MockAgent("reward_fn_code", "def compute_reward(*a): pass"),
    ]
    orchestrator = EnvironmentOrchestrator(agents=agents)
    await orchestrator.run(
        env_name="test_env",
        description="test",
        compiler_input=_compiler_input(),
    )
    assert (tmp_path / "test_env" / "app" / "main.py").exists()
    assert (tmp_path / "test_env" / "container_env.py").exists()
    assert (tmp_path / "test_env" / "custom" / "policies.yaml").exists()
    assert (tmp_path / "test_env" / "reward_fn.py").exists()


@pytest.mark.asyncio
async def test_orchestrator_calls_progress_callback(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path))
    progress = []

    async def on_progress(artifact_name: str, _value) -> None:
        progress.append(artifact_name)

    agents = [
        _MockAgent("app_code", {"main.py": "# app"}),
        _MockAgent("instrumented_code", {"main.py": "# instrumented"}, depends=["app_code"]),
        _MockAgent("state_bridge_code", "class ContainerForgeEnv: pass", depends=["instrumented_code"]),
        _MockAgent("policy_dsl", "policies: []"),
        _MockAgent("reward_fn_code", "def compute_reward(*a): pass"),
    ]
    orchestrator = EnvironmentOrchestrator(agents=agents, on_progress=on_progress)
    await orchestrator.run(env_name="test_env2", description="test", compiler_input=_compiler_input())
    assert set(progress) == {"app_code", "instrumented_code", "state_bridge_code", "policy_dsl", "reward_fn_code"}


@pytest.mark.asyncio
async def test_orchestrator_rejects_generated_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path))
    agents = [
        _MockAgent("app_code", {"../../escape.py": "# unsafe"}),
        _MockAgent("instrumented_code", {}, depends=["app_code"]),
        _MockAgent("state_bridge_code", "", depends=["instrumented_code"]),
        _MockAgent("policy_dsl", ""),
        _MockAgent("reward_fn_code", ""),
    ]
    with pytest.raises(ValueError, match="escapes configured root"):
        await EnvironmentOrchestrator(agents=agents).run(
            env_name="safe_env",
            description="test",
            compiler_input=_compiler_input(),
        )
    assert not (tmp_path / "escape.py").exists()


from forge.envgen.agents.reviewer import (
    GenerationReview, GenerationReviewError, ReviewIssue, ReviewSeverity,
)
from backend.app.services.env_orchestrator import enforce_generation_gates


def _approved() -> GenerationReview:
    return GenerationReview(approved=True)


def _rejected(category: str) -> GenerationReview:
    return GenerationReview(
        approved=False,
        issues=[ReviewIssue(severity=ReviewSeverity.ERROR, category=category, message="bad")],
    )


@pytest.mark.asyncio
async def test_gate_raises_when_correctness_report_rejects():
    bus = ArtifactBus()
    await bus.publish("review_report", _approved())
    await bus.publish("correctness_report", _rejected("wall_clock"))
    with pytest.raises(GenerationReviewError):
        enforce_generation_gates(bus)


@pytest.mark.asyncio
async def test_gate_passes_when_both_reports_approve():
    bus = ArtifactBus()
    await bus.publish("review_report", _approved())
    await bus.publish("correctness_report", _approved())
    enforce_generation_gates(bus)  # does not raise


@pytest.mark.asyncio
async def test_gate_raises_when_review_report_missing():
    bus = ArtifactBus()
    with pytest.raises(RuntimeError):
        enforce_generation_gates(bus)
