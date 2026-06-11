from __future__ import annotations
import json
from dataclasses import dataclass, field


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest_observation(obs, max_chars: int) -> str:
    """Compact, deterministic rendering of an observation.

    Small observations serialize whole; large ones collapse collections to
    item counts so the digest carries structure without the bulk.
    """
    full = _canonical(obs)
    if len(full) <= max_chars:
        return full
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs):
            value = obs[key]
            if isinstance(value, (dict, list)):
                parts.append(f'"{key}":<{len(value)} items>')
            else:
                parts.append(f'"{key}":{_canonical(value)}')
        return ("{" + ",".join(parts) + "}")[:max_chars]
    return full[:max_chars]


@dataclass
class ContextEntry:
    step_index: int
    action: dict
    observation_digest: str
    state_hash: str
    reward: float = 0.0
    is_error: bool = False

    def render(self) -> str:
        action_str = _canonical(self.action)
        if len(action_str) > 80:
            action_str = action_str[:77] + "..."
        return (
            f"step {self.step_index}: {action_str} -> "
            f"{self.state_hash[:12]} r={self.reward:+g}"
        )


@dataclass
class ContextDiagnosis:
    status: str  # "ok" | "stuck" | "context_limit_exceeded"
    stuck: bool
    context_limit_exceeded: bool
    detail: str = ""


class AgentContext:
    """Episode memory for an agent adapter: compact digest, stuck/limit
    diagnosis, and pruning so garbage never accumulates.

    record() each step; inject digest() into the agent prompt; consult
    diagnose() to tell a stuck agent (state stopped changing) apart from a
    blown token budget; prune() (automatic by default) drops error spam,
    revisited-state noise, and — if still over budget — the oldest history,
    always protecting the most recent `stuck_window` steps.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        stuck_window: int = 4,
        max_observation_chars: int = 500,
        auto_prune: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.stuck_window = stuck_window
        self.max_observation_chars = max_observation_chars
        self.auto_prune = auto_prune
        self.entries: list[ContextEntry] = []
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Recording & digest
    # ------------------------------------------------------------------

    def record(self, action: dict, observation, state_hash: str, reward: float = 0.0) -> None:
        self.entries.append(ContextEntry(
            step_index=self._step_counter,
            action=action,
            observation_digest=_digest_observation(observation, self.max_observation_chars),
            state_hash=state_hash,
            reward=reward,
            is_error=isinstance(observation, dict) and "error" in observation,
        ))
        self._step_counter += 1
        if self.auto_prune and self.token_estimate > self.max_tokens:
            self.prune()

    def digest(self) -> str:
        """Compact, deterministic context block for prompt injection."""
        if not self.entries:
            return ""
        lines = [entry.render() for entry in self.entries]
        lines.append(f"latest observation: {self.entries[-1].observation_digest}")
        return "\n".join(lines)

    @property
    def token_estimate(self) -> int:
        return len(self.digest()) // 4

    def clear(self) -> None:
        self.entries = []
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Diagnosis: stuck vs context limit
    # ------------------------------------------------------------------

    def diagnose(self) -> ContextDiagnosis:
        stuck = False
        detail = ""
        if len(self.entries) >= self.stuck_window:
            window = self.entries[-self.stuck_window:]
            if len({entry.state_hash for entry in window}) == 1:
                stuck = True
                detail = f"agent stuck: no state change in last {self.stuck_window} steps"

        over_budget = self.token_estimate > self.max_tokens
        if over_budget and not stuck:
            detail = (
                f"context limit exceeded: ~{self.token_estimate} tokens "
                f"> budget {self.max_tokens}"
            )

        status = "stuck" if stuck else "context_limit_exceeded" if over_budget else "ok"
        return ContextDiagnosis(
            status=status,
            stuck=stuck,
            context_limit_exceeded=over_budget,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self) -> int:
        """Drop garbage, then trim oldest history if still over budget.

        Garbage = error responses and zero-reward revisits of already-seen
        states. The most recent `stuck_window` entries are never pruned —
        they are what stuck-detection and the agent's immediate reasoning
        depend on.
        """
        before = len(self.entries)
        protected_from = max(0, len(self.entries) - self.stuck_window)

        kept: list[ContextEntry] = []
        seen_hashes: set[str] = set()
        for index, entry in enumerate(self.entries):
            protected = index >= protected_from
            garbage = entry.is_error or (entry.reward == 0 and entry.state_hash in seen_hashes)
            if protected or not garbage:
                kept.append(entry)
            seen_hashes.add(entry.state_hash)
        self.entries = kept

        while (
            self.token_estimate > self.max_tokens
            and len(self.entries) > self.stuck_window
        ):
            self.entries.pop(0)

        return before - len(self.entries)
