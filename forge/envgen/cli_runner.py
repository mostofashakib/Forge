from __future__ import annotations
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from forge.envgen.episode_base import (
    BaseEpisodeConfig,
    BaseEpisodeResult,
    TerminationMonitor,
)
from forge.envgen.objective import ObjectiveScorer
from forge.runtime.interaction import ComputerUse, ComputerUseSchema
from forge.runtime.snapshot import InvalidActionError
from forge.envgen.tiered_reward import (
    EndStateSpec,
    LoopDetector,
    LoopDetectorConfig,
    TieredRewardEngine,
    TrajectoryGrade,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CliEpisodeConfig(BaseEpisodeConfig):
    container_id: str
    command_timeout: float = 30.0
    # Loop-detector knobs (Tier 2 of the tiered reward).
    loop_repeat_threshold: int = 3
    loop_consecutive_failures: int = 5
    loop_window_size: int = 10


@dataclass(kw_only=True)
class CliEpisodeResult(BaseEpisodeResult):
    # Tiered-reward artifacts. None when grading was skipped (e.g. unit test
    # ran without a real container).
    end_state_spec: dict | None = None
    grade: dict | None = None

    def summary(self) -> dict:
        return {
            **super().summary(),
            "end_state_spec": self.end_state_spec,
            "grade": self.grade,
        }


class CliEpisodeRunner:
    def __init__(
        self,
        config: CliEpisodeConfig,
        scorer: ObjectiveScorer | None = None,
        reward_engine: TieredRewardEngine | None = None,
    ) -> None:
        self._cfg = config
        self._scorer = scorer or ObjectiveScorer()
        self._reward_engine = reward_engine or TieredRewardEngine()
        self._history: list[dict] = []

    def computer_use(self, schema: ComputerUseSchema | None = None) -> ComputerUse:
        """The ComputerUse contract a CLI environment grants the agent."""
        return ComputerUse(
            schema=schema or ComputerUseSchema(os="linux"),
            executor=lambda action: self._exec(action["command"]),
        )

    def _exec(self, command: str) -> dict:
        try:
            proc = subprocess.run(
                ["docker", "exec", self._cfg.container_id, "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=self._cfg.command_timeout,
            )
            return {
                "command": command,
                "stdout": proc.stdout[:4000],
                "stderr": proc.stderr[:1000],
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"command": command, "stdout": "", "stderr": "timed out", "exit_code": -1}
        except Exception as exc:
            return {"command": command, "stdout": "", "stderr": str(exc), "exit_code": -1}

    def _state(self, end_state_spec: EndStateSpec) -> dict:
        return {
            "environment_type": "cli",
            "objective_summary": end_state_spec.summary,
            "expected_steps": end_state_spec.expected_steps,
            "recent_history": self._history[-5:],
        }

    def run_episode(self, agent, episode_id: str | None = None, jsonl_path: Path | None = None) -> CliEpisodeResult:
        result = CliEpisodeResult()

        # Tier 1: derive an end-state spec from the natural-language objective.
        end_state_spec = self._reward_engine.plan_end_state(self._cfg.objective)
        result.end_state_spec = end_state_spec.to_dict()
        logger.info(
            "[cli-ep] end-state spec: %s (expected_steps=%d, %d assertions)",
            end_state_spec.summary, end_state_spec.expected_steps,
            len(end_state_spec.assertions),
        )

        # Tier 2: live loop / stuck-failure detector.
        loop_detector = LoopDetector(LoopDetectorConfig(
            repeat_threshold=self._cfg.loop_repeat_threshold,
            consecutive_failure_threshold=self._cfg.loop_consecutive_failures,
            window_size=self._cfg.loop_window_size,
        ))

        monitor = TerminationMonitor(self._cfg)
        early_termination: str | None = None
        computer_use = self.computer_use()

        for step_idx in range(self._cfg.max_steps):
            state = self._state(end_state_spec)

            try:
                command = agent.act(state=state, objective=self._cfg.objective)
            except Exception as exc:
                logger.warning("[cli-ep] step %d: agent.act failed: %s", step_idx, exc)
                command = "echo 'agent error'"

            try:
                exec_result = computer_use.execute({"action_type": "exec", "command": command})
            except InvalidActionError as exc:
                exec_result = {"command": command, "stdout": "", "stderr": exc.detail, "exit_code": -1}
            self._history.append(exec_result)

            score_state = {
                "command": command,
                "output": exec_result["stdout"][:500],
                "exit_code": exec_result["exit_code"],
                "step": step_idx + 1,
            }
            # Per-step score remains useful as a live signal in the UI even
            # though the FINAL reward now comes from the tiered grader.
            score = self._scorer.score(score_state, self._cfg.objective)

            step_record = {
                "step_index": step_idx,
                "command": command,
                "stdout": exec_result["stdout"],
                "stderr": exec_result["stderr"],
                "exit_code": exec_result["exit_code"],
                "objective_score": score,
                "reward": score,
            }
            result.steps.append(step_record)
            result.final_objective_score = score

            logger.info(
                "[cli-ep] step %02d/%d  cmd=%r  score=%.2f  exit=%d",
                step_idx + 1, self._cfg.max_steps, command[:50], score,
                exec_result["exit_code"],
            )

            # Tier 2: kill the episode if the agent is looping or stuck failing.
            loop_termination = loop_detector.observe(
                command, exec_result["exit_code"], exec_result["stdout"]
            )
            if loop_termination is not None:
                early_termination = loop_termination
                result.termination_reason = loop_termination
                break

            reason = monitor.observe(score)
            if reason is not None:
                result.termination_reason = reason
                break
        else:
            result.termination_reason = "max_steps"

        # Tier 3: grade the trajectory and replace the per-step running sum
        # with the tiered final reward.
        grade: TrajectoryGrade = self._reward_engine.grade(
            objective=self._cfg.objective,
            spec=end_state_spec,
            history=self._history,
            container_id=self._cfg.container_id,
            early_termination=early_termination,
        )
        result.grade = grade.to_dict()
        result.total_reward = grade.final_reward
        result.completed_at = datetime.now(timezone.utc)
        logger.info(
            "[cli-ep] graded: pass_rate=%.2f efficiency=%.2f partial=%.2f → reward=%.2f (%s)",
            grade.test_pass_rate, grade.efficiency_factor, grade.partial_credit,
            grade.final_reward, grade.reasoning,
        )
        if jsonl_path is not None:
            result.write_jsonl(jsonl_path)
        return result
