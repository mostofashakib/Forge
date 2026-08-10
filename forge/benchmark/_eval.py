"""Internal held-out evaluation for trained Forge policies.

External suites are intentionally out of scope here. The headline metric is
computed only from environments declared in an experiment's ``heldout_envs``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import pvariance
from typing import Callable

from forge.benchmark.compiled_tasks import CompiledTaskProvider, db_compiler_input_loader
from forge.benchmark.task_suite import Task
from forge.experiments import ExperimentConfig, RunResult
from forge.training.checkpoint import MANIFEST_NAME, PolicyCheckpoint, load_policy_agent


@dataclass(frozen=True)
class EpisodeOutcome:
    passed: bool
    reward: float
    reward_hacking: bool = False


EpisodeRunner = Callable[[Task, int, Path], EpisodeOutcome]


def evaluate_on_suite(
    model_path: str,
    suite: str,
    *,
    seed: int | None = None,
    runs_dir: str | Path = "runs",
    run_id: str | None = None,
    task_provider=None,
    episode_runner: EpisodeRunner | None = None,
    depth: int = 5,
) -> dict:
    """Evaluate ``policy_checkpoint.json`` only on configured held-out envs.

    ``suite`` is the path to a declarative experiment YAML. The legacy name is
    retained in the signature for CLI/API compatibility. External suite names
    are deliberately rejected.
    """
    experiment_path = Path(suite)
    config = ExperimentConfig.load(experiment_path)
    checkpoint_dir, checkpoint = _load_checkpoint(model_path)

    expected_config = config.model_dump(mode="json")
    if not checkpoint.experiment_config:
        raise ValueError(
            "checkpoint has no experiment metadata; retrain with forge train --experiment"
        )
    if checkpoint.experiment_config != expected_config:
        raise ValueError("checkpoint and evaluation experiment configs do not match")
    if checkpoint.base_model != config.base_model:
        raise ValueError("checkpoint base model does not match the experiment config")

    selected_seed = _resolve_seed(seed, checkpoint, config)
    resolved_run_id = run_id or checkpoint.run_id or (
        f"{experiment_path.stem}-seed-{selected_seed}"
    )

    if checkpoint.train_envs != config.train_envs:
        raise ValueError("checkpoint training environments do not match the experiment config")
    overlap = sorted(set(checkpoint.train_envs) & set(config.heldout_envs))
    if overlap:
        raise ValueError(
            f"refusing held-out evaluation on checkpoint training envs: {overlap}"
        )

    if task_provider is None:
        from backend.app.database import get_session_factory

        task_provider = CompiledTaskProvider(
            loader=db_compiler_input_loader(get_session_factory())
        )
    if episode_runner is None:
        episode_runner = _container_episode_runner(checkpoint_dir)

    outcomes: list[EpisodeOutcome] = []
    output_root = Path(runs_dir) / resolved_run_id / "eval"
    for env_name in config.heldout_envs:
        tasks = task_provider.tasks_for(domain=env_name, depth=depth)
        if not tasks:
            raise ValueError(f"held-out environment has no compiled tasks: {env_name}")
        for task in tasks:
            episode_path = output_root / env_name / task.name / f"seed_{selected_seed}.jsonl"
            outcomes.append(episode_runner(task, selected_seed, episode_path))

    if not outcomes:
        raise ValueError("the held-out split produced no evaluation episodes")

    pass_rate = sum(outcome.passed for outcome in outcomes) / len(outcomes)
    hacking_rate = sum(outcome.reward_hacking for outcome in outcomes) / len(outcomes)
    rewards = [outcome.reward for outcome in outcomes]
    reward_variance = pvariance(rewards) if len(rewards) > 1 else 0.0

    result_record = RunResult(
        config=config.model_dump(mode="json"),
        seed=selected_seed,
        heldout_pass_rate=pass_rate,
        reward_hacking_rate=hacking_rate,
        reward_variance=reward_variance,
    )
    result_path = result_record.save(runs_dir, resolved_run_id)
    return {
        **result_record.model_dump(),
        "run_id": resolved_run_id,
        "result_path": str(result_path),
        "num_eval_tasks": len(outcomes),
        # Compatibility for existing report consumers.
        "task_completion_rate": pass_rate,
    }


def _load_checkpoint(model_path: str) -> tuple[Path, PolicyCheckpoint]:
    path = Path(model_path)
    checkpoint_dir = path.parent if path.name == MANIFEST_NAME else path
    return checkpoint_dir, PolicyCheckpoint.load(checkpoint_dir)


def _resolve_seed(
    requested_seed: int | None,
    checkpoint: PolicyCheckpoint,
    config: ExperimentConfig,
) -> int:
    selected = requested_seed if requested_seed is not None else checkpoint.seed
    if selected is None:
        if len(config.seeds) != 1:
            raise ValueError(
                "experiment declares multiple seeds; pass seed= or use a seeded checkpoint"
            )
        selected = config.seeds[0]
    if selected not in config.seeds:
        raise ValueError(f"seed {selected} is not declared by the experiment")
    return selected


class _PolicyContainerAdapter:
    """Adapt a checkpoint runtime policy to the container runner action shape."""

    def __init__(self, policy_agent) -> None:
        self._policy_agent = policy_agent

    def act(self, state: dict, objective: str, available_actions: list[dict]) -> dict:
        tool_to_endpoint = {
            f"action_{index}": action["endpoint"]
            for index, action in enumerate(available_actions)
        }
        if not tool_to_endpoint:
            return {"endpoint": "/forge/state", "payload": {}}
        observation = {
            "objective": objective,
            "state": state,
            "available_actions": [
                {**action, "policy_tool": tool_name}
                for tool_name, action in zip(tool_to_endpoint, available_actions)
            ],
        }
        action = self._policy_agent.act(observation, frozenset(tool_to_endpoint))
        endpoint = tool_to_endpoint.get(str(action.get("type", "")), "")
        payload = {key: value for key, value in action.items() if key != "type"}
        return {"endpoint": endpoint, "payload": payload}


def _container_episode_runner(checkpoint_dir: Path) -> EpisodeRunner:
    from forge.envgen.episode_runner import ContainerEpisodeRunner, EpisodeConfig
    from forge.schema.state_schema import StateSchemaManifest
    from forge.settings import generated_envs_root

    policy = _PolicyContainerAdapter(load_policy_agent(checkpoint_dir))
    envs_root = generated_envs_root()

    def run(task: Task, seed: int, jsonl_path: Path) -> EpisodeOutcome:
        env_dir = envs_root / task.domain
        port_path = env_dir / "port"
        if not port_path.exists():
            raise FileNotFoundError(
                f"held-out environment is not running (missing {port_path})"
            )
        try:
            port = int(port_path.read_text(encoding="utf-8").strip())
        except ValueError as exc:
            raise ValueError(f"invalid environment port in {port_path}") from exc

        manifest = None
        manifest_path = env_dir / "state_schema.json"
        if manifest_path.exists():
            manifest = StateSchemaManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        episode_config = EpisodeConfig(
            base_url=f"http://localhost:{port}", objective=task.objective
        )
        with ContainerEpisodeRunner(episode_config, manifest=manifest) as runner:
            result = runner.run_episode(
                policy, jsonl_path=jsonl_path, seed=seed
            )
        passed = result.termination_reason == "success"
        return EpisodeOutcome(
            passed=passed,
            reward=result.total_reward,
            reward_hacking=_has_reward_hacking_pattern(result, passed),
        )

    return run


def _has_reward_hacking_pattern(result, passed: bool) -> bool:
    """Apply Forge's reward-hacking auditor to a successful eval episode."""
    if not passed:
        return False
    from forge.runtime.reward_hacking import RewardHackingAuditor
    from forge.runtime.verification import CheckResult, VerificationResult

    class EvalTrajectory:
        def __init__(self, steps) -> None:
            self.steps = steps
            self.events: list[dict] = []

        @property
        def step_count(self) -> int:
            return len(self.steps)

    trajectory = EvalTrajectory(result.steps)
    verification = VerificationResult.from_checks(
        "internal-heldout",
        [CheckResult(name="objective", passed=True, score=1.0)],
    )
    final_state = result.steps[-1].state_after if result.steps else {}
    report = RewardHackingAuditor(min_steps=1).audit(
        final_state,
        trajectory,
        {"objective": result.config.objective},
        verification,
    )
    return report.flagged
