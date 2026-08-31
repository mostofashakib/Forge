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
from forge.grading_provenance import (
    require_independent_grader,
    resolve_grading_provenance,
)
from forge.reward_presets import reward_preset_spec
from forge.settings import determinism_enabled
from forge.runtime.policy_loader import load_policy_agent
from forge.training.checkpoint import MANIFEST_NAME, PolicyCheckpoint


@dataclass(frozen=True)
class EpisodeOutcome:
    passed: bool
    reward: float
    reward_hacking: bool = False
    # Verdicts a model issued while grading this episode, counted by the runner.
    llm_verdicts: int = 0
    # True when the verdict jury could not agree. Such an episode leaves the
    # pass-rate denominator rather than being counted as a failure.
    indeterminate: bool = False


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
    llm_graded: bool | None = None,
    verdict_jury=None,
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
        episode_runner = _container_episode_runner(
            checkpoint_dir, config.reward_preset, verdict_jury=verdict_jury
        )

    # Resolve who generated and who graded before spending any episodes: a
    # contaminated grader invalidates the run, so fail before the work, not
    # after. The runner is asked first because it knows what it will actually
    # do — the reward preset only describes the verifier layers, and the
    # container runner issues an LLM verdict on every step under any preset.
    if llm_graded is None:
        llm_graded = getattr(episode_runner, "issues_llm_verdicts", None)
    if llm_graded is None:
        llm_graded = reward_preset_spec(config.reward_preset).issues_llm_verdict
    provenance = resolve_grading_provenance(llm_graded=llm_graded)
    if config.require_grader_independence:
        require_independent_grader(provenance)

    outcomes: list[EpisodeOutcome] = []
    reward_groups: list[list[float]] = []
    output_root = Path(runs_dir) / resolved_run_id / "eval"
    for env_name in config.heldout_envs:
        tasks = task_provider.tasks_for(domain=env_name, depth=depth)
        if not tasks:
            raise ValueError(f"held-out environment has no compiled tasks: {env_name}")
        for task in tasks:
            task_rewards: list[float] = []
            for repeat in range(config.determinism_repeats):
                episode_path = (
                    output_root / env_name / task.name
                    / f"seed_{selected_seed}_repeat_{repeat}.jsonl"
                )
                outcome = episode_runner(task, selected_seed, episode_path)
                if (
                    reward_preset_spec(config.reward_preset).auditor_enabled
                    and outcome.reward_hacking
                ):
                    outcome = EpisodeOutcome(
                        passed=False, reward=0.0, reward_hacking=True
                    )
                outcomes.append(outcome)
                task_rewards.append(outcome.reward)
            reward_groups.append(task_rewards)

    if not outcomes:
        raise ValueError("the held-out split produced no evaluation episodes")

    abstention_rate = sum(o.indeterminate for o in outcomes) / len(outcomes)
    if abstention_rate > config.max_abstention_rate:
        # Refuse to publish a pass rate computed from what is left. A jury that
        # cannot decide this share of its cases is a broken instrument, and the
        # remaining episodes are a biased sample of the ones it found easy.
        raise ValueError(
            f"abstention rate {abstention_rate:.2f} exceeds the configured "
            f"maximum {config.max_abstention_rate:.2f}; the verdict jury could "
            "not decide enough episodes for the pass rate to mean anything"
        )

    decided = [outcome for outcome in outcomes if not outcome.indeterminate]
    if not decided:
        raise ValueError("every evaluated episode was indeterminate")

    pass_rate = sum(outcome.passed for outcome in decided) / len(decided)
    hacking_rate = sum(outcome.reward_hacking for outcome in decided) / len(decided)
    task_variances = [pvariance(rewards) for rewards in reward_groups]
    reward_variance = sum(task_variances) / len(task_variances)

    # Replace the declared grading mode with what the run actually did. This
    # raises rather than writing a record that understates model involvement.
    provenance = provenance.with_observed_verdicts(
        sum(outcome.llm_verdicts for outcome in outcomes)
    )

    result_record = RunResult(
        config=config.model_dump(mode="json"),
        seed=selected_seed,
        determinism="on" if determinism_enabled() else "off",
        heldout_pass_rate=pass_rate,
        reward_hacking_rate=hacking_rate,
        reward_variance=reward_variance,
        abstention_rate=abstention_rate,
        grading=provenance.as_record(),
    )
    result_path = result_record.save(runs_dir, resolved_run_id)
    return {
        **result_record.model_dump(),
        "run_id": resolved_run_id,
        "result_path": str(result_path),
        "num_eval_tasks": len(reward_groups),
        "num_eval_episodes": len(outcomes),
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


def _container_episode_runner(
    checkpoint_dir: Path, reward_preset, verdict_jury=None
) -> EpisodeRunner:
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
        verdict = resolve_verdict(result, task, reward_preset, jury=verdict_jury)
        preset = reward_preset_spec(reward_preset)
        reward = (
            1.0 if preset.binary_final_state and verdict.passed
            else result.total_reward
        )
        if preset.binary_final_state and not verdict.passed:
            reward = 0.0
        return EpisodeOutcome(
            passed=verdict.passed,
            reward=reward,
            reward_hacking=_has_reward_hacking_pattern(result, verdict.passed),
            # ObjectiveScorer still runs every step to shape the reward, so the
            # count is honest even when the pass/fail came from a computed
            # check rather than a model.
            llm_verdicts=result.llm_verdicts,
            indeterminate=verdict.indeterminate,
        )

    # ContainerEpisodeRunner scores every step with ObjectiveScorer to shape the
    # per-step reward, so a model issues verdicts on this path regardless of how
    # pass/fail is decided.
    run.issues_llm_verdicts = True
    return run


class _EpisodeTrajectory:
    """Adapt a container episode result to the trajectory shape verifiers read."""

    def __init__(self, result) -> None:
        self.steps = list(getattr(result, "steps", []))
        self.events = list(getattr(result, "events", []) or [])

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def actions(self) -> list[dict]:
        return [getattr(step, "action", {}) or {} for step in self.steps]


def structural_verdict(result, task: Task, reward_preset) -> bool | None:
    """Grade an episode against its compiled success and failure conditions.

    Returns ``None`` when there is no structural ground truth to grade against —
    the task carries no compiled template, declares no conditions, or the preset
    enables no structural layer. Absent ground truth is *unknown*, never success:
    the caller decides what to do with an ungradeable episode rather than
    inheriting a free pass.
    """
    template = getattr(task, "template", None)
    if template is None or not template.success_conditions:
        return None

    from forge.runtime.verifier_composer import VerifierComposer

    composer = VerifierComposer(reward_preset)
    try:
        verifier = composer.compose(template, verifier_id=task.name)
    except ValueError:
        # The preset demands a condition type this task does not declare.
        return None
    if not verifier.has_structural_checks:
        return None

    final_state = (
        getattr(result.steps[-1], "state_after", {}) if result.steps else {}
    )
    verification = verifier(
        final_state,
        _EpisodeTrajectory(result),
        {"name": task.name, "objective": task.objective},
    )
    return bool(verification.passed)


@dataclass(frozen=True)
class EpisodeVerdict:
    """How an episode was decided, and by what."""

    passed: bool
    source: str
    indeterminate: bool = False

    @property
    def llm_derived(self) -> bool:
        """True when a model, not a computed check, produced this verdict."""
        return self.source != "structural"


def resolve_verdict(
    result, task: Task, reward_preset, jury=None
) -> EpisodeVerdict:
    """Decide an episode, preferring computed ground truth over model opinion.

    Order matters. A structural verdict against the environment's own compiled
    conditions is a function of the recorded trajectory, so it is preferred
    whenever it exists. Only when a task carries no compiled ground truth does
    this fall back to the run's termination reason — which the objective scorer
    drives, and which is therefore recorded as LLM-derived.

    A verdict jury, when supplied, votes on top of that verdict; a jury that
    cannot agree yields an indeterminate episode rather than a coin flip.
    """
    structural = structural_verdict(result, task, reward_preset)
    if structural is not None:
        verdict = EpisodeVerdict(passed=structural, source="structural")
    else:
        verdict = EpisodeVerdict(
            passed=result.termination_reason == "success",
            source="termination_reason",
        )

    if jury is None:
        return verdict

    outcome = jury.deliberate({
        "task": task.name,
        "objective": task.objective,
        "passed": verdict.passed,
        "steps": len(getattr(result, "steps", [])),
    })
    if outcome.indeterminate:
        # Undecided is not failed. The caller drops it from the denominator.
        return EpisodeVerdict(passed=False, source=verdict.source, indeterminate=True)
    return EpisodeVerdict(passed=bool(outcome.decision), source=verdict.source)


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
