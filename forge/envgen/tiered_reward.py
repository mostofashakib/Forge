"""Tiered reward engine for CLI agent episodes.

The reward for a CLI episode is composed in three tiers:

1. **Spec generation (once, at episode start).** An LLM converts the natural-
   language objective into an `EndStateSpec`: a list of bash assertions that
   exit 0 iff the goal is achieved, plus an estimate of the optimal step count.
   These assertions are the source of ground truth — they're the same kind of
   black-box test a human grader would write.

2. **Live loop detection (per step).** A `LoopDetector` watches a sliding window
   of (command, exit_code, stdout-hash) triples. If the agent repeats the same
   action+result N times, or fails (non-zero exit) M times in a row without any
   intervening progress, the episode is killed with `reward = 0`. This stops
   reward leakage from agents stuck thrashing.

3. **Trajectory grading (once, at episode end).** Run the assertions inside the
   container. Combine three signals into the final reward:

       final = efficiency_factor × reward_from_tests

   - `efficiency_factor = clip(expected_steps / actual_steps, 0.5, 1.0)` — a
     mild penalty for taking 2× the expected steps, never punitive enough to
     wipe out a real success.
   - `reward_from_tests`:
       * all assertions pass → `1.0`
       * some pass → `pass_rate` (e.g. 6/10 assertions = 0.6)
       * none pass → LLM grades the trajectory and assigns 0.0–0.4 partial
         credit based on how close the agent got.

This module owns the LLM calls and the assertion-running logic. The
`CliEpisodeRunner` calls into it but remains responsible for the actual
`docker exec` loop.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import BaseModel, Field

from forge.extraction.llm_client import AnthropicClient, LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas — what the LLM returns
# ---------------------------------------------------------------------------

class _SpecAssertion(BaseModel):
    description: str = Field(description="What this assertion checks (one short sentence).")
    command: str = Field(
        description=(
            "A bash one-liner that exits 0 iff the assertion holds, non-zero otherwise. "
            "Run via `docker exec <id> bash -c '<command>'`. No interactive prompts. "
            "Use `[[ … ]]` / `test` / `grep -q` / `diff -q` style checks."
        ),
    )


class _EndStateSpecLLM(BaseModel):
    summary: str = Field(description="One-sentence description of the desired end state.")
    expected_steps: int = Field(
        description="Rough estimate of the optimal number of agent steps to achieve this objective.",
        ge=1,
    )
    assertions: list[_SpecAssertion] = Field(
        description="Independent shell-level checks. 3-8 is a good range. Each must be self-contained.",
        min_length=1,
    )


class _PartialCreditLLM(BaseModel):
    score: float = Field(description="Partial credit between 0.0 and 0.4.", ge=0.0, le=0.4)
    reasoning: str = Field(description="One short sentence explaining the score.")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EndStateSpec:
    """The end-state contract derived from the natural-language objective."""
    summary: str
    expected_steps: int
    assertions: list[dict]  # each {"description": ..., "command": ...}

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "expected_steps": self.expected_steps,
            "assertions": list(self.assertions),
        }


@dataclass
class AssertionResult:
    description: str
    command: str
    passed: bool
    exit_code: int
    stderr: str

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "command": self.command,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "stderr": self.stderr[:500],
        }


@dataclass
class TrajectoryGrade:
    """Final breakdown — what made up the reward, exposed for inspection."""
    final_reward: float
    test_pass_rate: float
    efficiency_factor: float
    partial_credit: float
    test_results: list[AssertionResult] = field(default_factory=list)
    reasoning: str = ""
    expected_steps: int = 0
    actual_steps: int = 0
    early_termination: str | None = None

    def to_dict(self) -> dict:
        return {
            "final_reward": self.final_reward,
            "test_pass_rate": self.test_pass_rate,
            "efficiency_factor": self.efficiency_factor,
            "partial_credit": self.partial_credit,
            "test_results": [r.to_dict() for r in self.test_results],
            "reasoning": self.reasoning,
            "expected_steps": self.expected_steps,
            "actual_steps": self.actual_steps,
            "early_termination": self.early_termination,
        }


# ---------------------------------------------------------------------------
# Loop / dead-end detection
# ---------------------------------------------------------------------------

@dataclass
class LoopDetectorConfig:
    # Same (command, exit_code, stdout) seen this many times → loop.
    repeat_threshold: int = 3
    # This many consecutive failed (non-zero exit_code) steps → stuck.
    consecutive_failure_threshold: int = 5
    # Sliding-window size for the repeat detector.
    window_size: int = 10


class LoopDetector:
    """Tracks recent step fingerprints to detect loops and stuck-failure runs.

    Lightweight — purely local heuristics, no LLM cost. The two heuristics
    catch most stuck-agent failure modes:
      * the same command+output recurring (agent retrying its own bad guess)
      * a long uninterrupted streak of non-zero exits (agent thrashing without
        ever producing useful side effects)
    """

    def __init__(self, config: LoopDetectorConfig | None = None) -> None:
        self._cfg = config or LoopDetectorConfig()
        self._window: deque[str] = deque(maxlen=self._cfg.window_size)
        self._consecutive_failures = 0

    @staticmethod
    def _fingerprint(command: str, exit_code: int, stdout: str) -> str:
        return hashlib.sha256(
            f"{command}\x00{exit_code}\x00{stdout[:512]}".encode("utf-8")
        ).hexdigest()[:16]

    def observe(self, command: str, exit_code: int, stdout: str) -> str | None:
        """Record a step. Returns a non-None termination reason if the agent
        looks stuck and the episode should be killed."""
        fp = self._fingerprint(command, exit_code, stdout)
        self._window.append(fp)
        # Repeat detector
        repeats = sum(1 for f in self._window if f == fp)
        if repeats >= self._cfg.repeat_threshold:
            return "loop_detected"
        # Consecutive-failure detector
        if exit_code != 0:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        if self._consecutive_failures >= self._cfg.consecutive_failure_threshold:
            return "stuck_failing"
        return None


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = (
    "You design grading rubrics for CLI agent episodes.\n"
    "\n"
    "Given a natural-language objective for an Ubuntu container, produce a\n"
    "machine-checkable end-state spec consisting of:\n"
    "  - a one-sentence summary of what 'done' looks like\n"
    "  - an integer estimate of the optimal number of shell commands needed\n"
    "  - 3-8 independent bash assertions that exit 0 iff the assertion holds\n"
    "\n"
    "Each assertion is run as `docker exec <id> bash -c '<command>'`. Treat\n"
    "non-zero exit codes as failure. Examples of good assertions:\n"
    "  - `[[ -f /tmp/foo.txt ]]`                                   — file exists\n"
    "  - `grep -q 'expected line' /etc/hosts`                      — content match\n"
    "  - `[[ \"$(wc -l < /tmp/log)\" -ge 10 ]]`                    — count check\n"
    "  - `python3 -c \"import json,sys;json.load(open('a.json'))\"` — file is valid JSON\n"
    "\n"
    "Make assertions independent — passing one shouldn't depend on another.\n"
    "Do NOT generate destructive checks (no `rm`, no shutdown).\n"
    "Call the extract tool with the spec."
)

_PARTIAL_CREDIT_SYSTEM = (
    "You assess a CLI agent's failed trajectory and assign partial credit between\n"
    "0.0 and 0.4. The agent did not satisfy any of the end-state assertions, but\n"
    "we want to reward genuine progress.\n"
    "\n"
    "Score the trajectory:\n"
    "  0.0  — agent took no actions toward the objective, or only invalid commands\n"
    "  0.1  — agent attempted the right kind of work but made fundamental errors\n"
    "  0.2  — agent partially completed prerequisite work but not the goal itself\n"
    "  0.3  — agent was very close to passing, e.g. wrong file path or off-by-one\n"
    "  0.4  — agent likely satisfies the spirit of the objective even if assertions\n"
    "         technically fail (cap — fully-correct trajectories should pass tests)\n"
    "\n"
    "Be skeptical. Most failed trajectories deserve 0.0–0.2. Reserve 0.3-0.4 for\n"
    "trajectories where the agent clearly understood the task.\n"
    "\n"
    "Call the extract tool with score and reasoning."
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class TieredRewardConfig:
    # Lower bound on the efficiency multiplier: even a very inefficient agent
    # that nonetheless succeeds should not see its reward zeroed out by step
    # count alone. 0.5 means "at most 2× penalty for inefficiency".
    min_efficiency: float = 0.5
    # Above this fraction of failed assertions, a "partial" trajectory falls
    # through to the grader instead of being scored from pass-rate alone.
    # (i.e. only invoke the grader when `pass_rate == 0`.)
    llm_grade_when_zero_pass: bool = True
    # Per-assertion exec timeout, seconds.
    assertion_timeout: float = 15.0
    # Scoring methods for partial credit when no assertions pass.
    # Multiple methods are averaged together.
    # "llm"        — LLM-as-judge (Claude Haiku, default)
    # "embeddings" — cosine similarity via sentence-transformers
    # "rouge"      — ROUGE-L lexical overlap
    # "bleu"       — BLEU n-gram precision
    partial_credit_methods: list[str] = field(default_factory=lambda: ["llm"])


class TieredRewardEngine:
    """Three-tier grader — spec generation, loop detection, trajectory grading."""

    def __init__(
        self,
        client: LLMClient | None = None,
        config: TieredRewardConfig | None = None,
    ) -> None:
        self._client = client or AnthropicClient(
            model="claude-haiku-4-5-20251001", max_tokens=1024
        )
        self._cfg = config or TieredRewardConfig()

    # -- Tier 1: planning ---------------------------------------------------

    def plan_end_state(self, objective: str) -> EndStateSpec:
        """Convert the objective into an executable spec.

        Falls back to a minimal spec on LLM failure so the episode can still
        run (it just won't get test-based grading — only LLM partial credit
        at the end).
        """
        try:
            result: _EndStateSpecLLM = self._client.extract(
                system=_PLANNER_SYSTEM,
                user=f"Objective:\n{objective}",
                schema=_EndStateSpecLLM,
            )
            return EndStateSpec(
                summary=result.summary,
                expected_steps=max(1, int(result.expected_steps)),
                assertions=[
                    {"description": a.description, "command": a.command}
                    for a in result.assertions
                ],
            )
        except Exception as exc:
            logger.warning("[tiered-reward] plan_end_state LLM failed: %s", exc)
            return EndStateSpec(
                summary=objective,
                expected_steps=10,
                assertions=[],
            )

    # -- Tier 3: trajectory grading ----------------------------------------

    def grade(
        self,
        objective: str,
        spec: EndStateSpec,
        history: Sequence[dict],
        container_id: str,
        early_termination: str | None = None,
    ) -> TrajectoryGrade:
        """Run the spec's assertions and produce the final reward breakdown.

        `early_termination` is set when Tier 2 (loop detector) killed the run.
        In that case we short-circuit to reward = 0 — the agent demonstrated
        no meaningful progress, so don't waste an LLM call on it.
        """
        actual_steps = len(history)

        if early_termination in ("loop_detected", "stuck_failing"):
            return TrajectoryGrade(
                final_reward=0.0,
                test_pass_rate=0.0,
                efficiency_factor=0.0,
                partial_credit=0.0,
                test_results=[],
                reasoning=f"Episode killed early: {early_termination}.",
                expected_steps=spec.expected_steps,
                actual_steps=actual_steps,
                early_termination=early_termination,
            )

        # Run the assertions. If the LLM produced no assertions (planner
        # fallback), pass_rate stays 0 and we lean entirely on partial credit.
        test_results = self._run_assertions(spec.assertions, container_id)
        passed = sum(1 for r in test_results if r.passed)
        total = len(test_results) or 1
        pass_rate = passed / total if test_results else 0.0

        efficiency = self._efficiency(spec.expected_steps, actual_steps)

        # Decide on the reward source.
        if test_results and pass_rate == 1.0:
            base = 1.0
            partial_credit = 0.0
            reasoning = "All end-state assertions passed."
        elif pass_rate > 0.0:
            base = pass_rate
            partial_credit = 0.0
            reasoning = f"{passed}/{total} end-state assertions passed."
        else:
            # No tests passed — defer to the configured grader(s).
            if self._cfg.llm_grade_when_zero_pass:
                partial_credit = self._grade_partial_combined(objective, spec, history)
                method_label = "+".join(self._cfg.partial_credit_methods)
            else:
                partial_credit = 0.0
                method_label = "none"
            base = partial_credit
            reasoning = (
                f"No assertions passed; {method_label} partial credit = {partial_credit:.2f}."
                if test_results
                else f"No assertions generated; {method_label} partial credit = {partial_credit:.2f}."
            )

        final = max(0.0, min(1.0, efficiency * base))
        return TrajectoryGrade(
            final_reward=final,
            test_pass_rate=pass_rate,
            efficiency_factor=efficiency,
            partial_credit=partial_credit,
            test_results=test_results,
            reasoning=reasoning,
            expected_steps=spec.expected_steps,
            actual_steps=actual_steps,
            early_termination=None,
        )

    # -- Helpers -----------------------------------------------------------

    def _efficiency(self, expected: int, actual: int) -> float:
        """min_efficiency ≤ expected/actual ≤ 1.0.

        Agents finishing in ≤ expected steps get the full multiplier (1.0).
        Agents taking 2× expected get ~0.5. Agents taking 10× expected
        bottom out at `min_efficiency`.
        """
        if actual <= 0:
            return 0.0
        ratio = expected / actual
        return max(self._cfg.min_efficiency, min(1.0, ratio))

    def _run_assertions(
        self, assertions: list[dict], container_id: str
    ) -> list[AssertionResult]:
        results: list[AssertionResult] = []
        for a in assertions:
            command = a.get("command", "")
            description = a.get("description", "")
            try:
                proc = subprocess.run(
                    ["docker", "exec", container_id, "bash", "-c", command],
                    capture_output=True,
                    text=True,
                    timeout=self._cfg.assertion_timeout,
                )
                results.append(AssertionResult(
                    description=description,
                    command=command,
                    passed=proc.returncode == 0,
                    exit_code=proc.returncode,
                    stderr=proc.stderr,
                ))
            except subprocess.TimeoutExpired:
                results.append(AssertionResult(
                    description=description,
                    command=command,
                    passed=False,
                    exit_code=-1,
                    stderr=f"timed out after {self._cfg.assertion_timeout}s",
                ))
            except Exception as exc:
                results.append(AssertionResult(
                    description=description,
                    command=command,
                    passed=False,
                    exit_code=-1,
                    stderr=str(exc)[:500],
                ))
        return results

    def _trajectory_to_text(self, history: Sequence[dict]) -> str:
        """Compact text representation of a trajectory for ML scoring."""
        lines = []
        for step in history[-30:]:
            cmd = (step.get("command") or "")[:120]
            stdout = (step.get("stdout") or "").strip().split("\n", 1)[0][:160]
            lines.append(f"$ {cmd}\n{stdout}")
        return "\n".join(lines)

    def _grade_partial_combined(
        self, objective: str, spec: EndStateSpec, history: Sequence[dict]
    ) -> float:
        """Average partial credit across all configured methods.

        LLM returns 0.0–0.4 directly. ML methods return 0.0–1.0 similarity
        which is scaled to 0.0–0.4 to match the same range.
        """
        from forge.envgen.ml_reward import build_scorer
        scores: list[float] = []
        reference = f"{objective}\n{spec.summary}"
        candidate = self._trajectory_to_text(history)

        for method in self._cfg.partial_credit_methods:
            if method == "llm":
                scores.append(self._grade_partial(objective, spec, history))
            else:
                scorer = build_scorer(method)
                if scorer is None:
                    continue
                try:
                    raw = scorer.score(reference, candidate)
                    scores.append(max(0.0, min(0.4, raw * 0.4)))
                except Exception as exc:
                    logger.warning("[tiered-reward] ML partial credit (%s) failed: %s", method, exc)

        return sum(scores) / len(scores) if scores else 0.0

    def _grade_partial(
        self, objective: str, spec: EndStateSpec, history: Sequence[dict]
    ) -> float:
        """LLM-based partial credit when no assertion passed."""
        try:
            # Compact the trajectory: command + exit + first line of stdout.
            trajectory_summary = []
            for i, step in enumerate(history[-30:]):  # cap context
                cmd = (step.get("command") or "")[:120]
                stdout = (step.get("stdout") or "").strip().split("\n", 1)[0][:160]
                exit_code = step.get("exit_code", "?")
                trajectory_summary.append(f"  {i+1}. $ {cmd}\n     exit={exit_code} | {stdout}")
            user = (
                f"Objective:\n{objective}\n\n"
                f"Expected end state:\n{spec.summary}\n\n"
                f"Required end-state assertions (none passed):\n"
                + "\n".join(f"  - {a['description']}" for a in spec.assertions)
                + "\n\n"
                f"Agent trajectory ({len(history)} steps total, last 30 shown):\n"
                + "\n".join(trajectory_summary)
            )
            result: _PartialCreditLLM = self._client.extract(
                system=_PARTIAL_CREDIT_SYSTEM, user=user, schema=_PartialCreditLLM,
            )
            return max(0.0, min(0.4, float(result.score)))
        except Exception as exc:
            logger.warning("[tiered-reward] partial-credit LLM failed: %s", exc)
            return 0.0
