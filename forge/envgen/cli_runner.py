from __future__ import annotations
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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


@dataclass
class CliEpisodeConfig:
    container_id: str
    objective: str
    max_steps: int = 30
    divergence_threshold: float = 0.2
    consecutive_below_threshold: int = 3
    dead_end_patience: int = 5
    success_threshold: float = 0.9
    command_timeout: float = 30.0
    # Loop-detector knobs (Tier 2 of the tiered reward).
    loop_repeat_threshold: int = 3
    loop_consecutive_failures: int = 5
    loop_window_size: int = 10


@dataclass
class CliEpisodeResult:
    steps: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    final_objective_score: float = 0.0
    termination_reason: str = "unknown"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    # Tiered-reward artifacts. None when grading was skipped (e.g. unit test
    # ran without a real container).
    end_state_spec: dict | None = None
    grade: dict | None = None

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(s) for s in self.steps]
        summary = {
            "type": "episode_summary",
            "total_steps": len(self.steps),
            "total_reward": self.total_reward,
            "final_objective_score": self.final_objective_score,
            "termination_reason": self.termination_reason,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "end_state_spec": self.end_state_spec,
            "grade": self.grade,
        }
        lines.append(json.dumps(summary))
        path.write_text("\n".join(lines), encoding="utf-8")


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

        below_threshold_count = 0
        score_window: list[float] = []
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
            score_window.append(score)

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

            if score >= self._cfg.success_threshold:
                result.termination_reason = "success"
                break

            if len(score_window) >= self._cfg.dead_end_patience:
                recent = score_window[-self._cfg.dead_end_patience:]
                if len(set(round(s, 2) for s in recent)) == 1:
                    result.termination_reason = "dead_end"
                    break

            if score < self._cfg.divergence_threshold:
                below_threshold_count += 1
            else:
                below_threshold_count = 0
            if below_threshold_count >= self._cfg.consecutive_below_threshold:
                result.termination_reason = "diverged"
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
