from __future__ import annotations
import json
import random

from pydantic import BaseModel

from forge.extraction.llm_client import LLMClient, get_client
from forge.runtime.errors import AgentError


_CLI_SYSTEM = (
    "You are an autonomous agent controlling a Linux terminal to achieve an objective.\n"
    "You will receive:\n"
    "  - The objective you must accomplish\n"
    "  - The recent command history (commands and their output)\n"
    "\n"
    "Return the single best shell command to make progress toward the objective.\n"
    "Use common Linux tools (ls, cat, grep, mkdir, touch, echo, pip, python, etc.).\n"
    "Keep commands focused. If the objective is already achieved, run a no-op like 'echo done'.\n"
    "Call the extract tool with the command and your one-sentence reasoning."
)


class _CommandSchema(BaseModel):
    command: str
    reasoning: str


class LLMCliAgent:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client(max_tokens=256)

    def act(self, state: dict, objective: str) -> str:
        history_text = json.dumps(state.get("recent_history", []), indent=2)
        user = (
            f"Objective: {objective}\n\n"
            f"Recent command history:\n{history_text}"
        )
        result = self._client.extract(system=_CLI_SYSTEM, user=user, schema=_CommandSchema)
        return result.command


class RandomCliAgent:
    _COMMANDS = [
        "ls -la", "pwd", "whoami", "echo hello", "cat /etc/os-release",
        "df -h", "free -m", "uname -a", "env | head -10", "ps aux | head -15",
    ]

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, state: dict, objective: str) -> str:
        return self._rng.choice(self._COMMANDS)


class ReplayCliAgent:
    """Replays a pre-recorded command sequence from a synthetic trajectory manifest."""

    def __init__(self, commands: list[str]) -> None:
        self._commands = commands
        self._step = 0

    def act(self, state: dict, objective: str) -> str:
        if self._step < len(self._commands):
            cmd = self._commands[self._step]
            self._step += 1
            return cmd
        return "echo 'trajectory complete'"


def make_cli_agent(agent_id: str, seed: int | None = None):
    if agent_id == "random":
        return RandomCliAgent(seed=seed)
    if agent_id == "llm" or agent_id.startswith("llm:"):
        model = agent_id[4:] if agent_id.startswith("llm:") else None
        return LLMCliAgent(get_client(max_tokens=256, model=model))
    raise AgentError(f"Unknown CLI agent id: {agent_id!r}. Use 'random', 'llm', or 'llm:<model>'.")
