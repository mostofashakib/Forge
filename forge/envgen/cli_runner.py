from __future__ import annotations
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from forge.envgen.objective import ObjectiveScorer

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


@dataclass
class CliEpisodeResult:
    steps: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    final_objective_score: float = 0.0
    termination_reason: str = "unknown"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

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
        }
        lines.append(json.dumps(summary))
        path.write_text("\n".join(lines), encoding="utf-8")


class CliEpisodeRunner:
    def __init__(self, config: CliEpisodeConfig, scorer: ObjectiveScorer | None = None) -> None:
        self._cfg = config
        self._scorer = scorer or ObjectiveScorer()
        self._history: list[dict] = []

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

    def _state(self) -> dict:
        return {
            "environment_type": "cli",
            "recent_history": self._history[-5:],
        }

    def run_episode(self, agent, episode_id: str | None = None, jsonl_path: Path | None = None) -> CliEpisodeResult:
        result = CliEpisodeResult()
        below_threshold_count = 0
        score_window: list[float] = []

        for step_idx in range(self._cfg.max_steps):
            state = self._state()

            try:
                command = agent.act(state=state, objective=self._cfg.objective)
            except Exception as exc:
                logger.warning("[cli-ep] step %d: agent.act failed: %s", step_idx, exc)
                command = "echo 'agent error'"

            exec_result = self._exec(command)
            self._history.append(exec_result)

            score_state = {
                "command": command,
                "output": exec_result["stdout"][:500],
                "exit_code": exec_result["exit_code"],
                "step": step_idx + 1,
            }
            score = self._scorer.score(score_state, self._cfg.objective)
            result.total_reward += score
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
                "[cli-ep] step %02d/%d  cmd=%r  score=%.2f",
                step_idx + 1, self._cfg.max_steps, command[:50], score,
            )

            # Stopping conditions
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

        result.completed_at = datetime.now(timezone.utc)
        if result.steps:
            result.total_reward = result.total_reward / len(result.steps)
        if jsonl_path is not None:
            result.write_jsonl(jsonl_path)
        return result
