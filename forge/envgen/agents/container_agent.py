from __future__ import annotations
import json
import random
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from forge.extraction.llm_client import LLMClient, get_client
from forge.runtime.errors import AgentError
from forge.envgen.config import envgen_config


@runtime_checkable
class ContainerAgentBase(Protocol):
    def act(
        self,
        state: dict,
        objective: str,
        available_actions: list[dict],
    ) -> dict:
        """Return {"endpoint": "/some/path", "payload": {...}}."""
        ...


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

class _ActionSchema(BaseModel):
    endpoint: str
    payload: dict
    reasoning: str


_AGENT_SYSTEM = (
    "You are an autonomous agent controlling a web application to achieve an objective.\n"
    "You will receive:\n"
    "  - The objective you must accomplish\n"
    "  - The current application state as JSON\n"
    "  - A list of available action endpoints with their request schemas\n"
    "\n"
    "Pick the single best action that makes the most progress toward the objective.\n"
    "If the objective is already achieved, call any no-op action.\n"
    "Populate the payload with realistic values that match the request schema.\n"
    "Call the extract tool with the chosen endpoint, payload, and your one-sentence reasoning."
)


class ContainerAgentPrompts:
    SYSTEM = _AGENT_SYSTEM


class LLMContainerAgent:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client(max_tokens=envgen_config().action_llm_tokens)

    def act(
        self,
        state: dict,
        objective: str,
        available_actions: list[dict],
    ) -> dict:
        state_text = json.dumps(state, indent=2)
        if len(state_text) > 3000:
            state_text = state_text[:3000] + "\n... (truncated)"

        # Trim action schemas to keep prompt small
        trimmed = [
            {
                "endpoint": a["endpoint"],
                "description": a.get("description", ""),
                "parameters": a.get("request_schema", {}).get("properties", {}),
            }
            for a in available_actions
        ]

        user = (
            f"Objective: {objective}\n\n"
            f"Current state:\n{state_text}\n\n"
            f"Available actions:\n{json.dumps(trimmed, indent=2)}"
        )
        result = self._client.extract(
            system=ContainerAgentPrompts.SYSTEM, user=user, schema=_ActionSchema
        )
        # Include reasoning so it's stored in the trajectory JSONL and visible
        # in the trajectory viewer for post-hoc analysis and debugging.
        return {"endpoint": result.endpoint, "payload": result.payload, "reasoning": result.reasoning}


# ---------------------------------------------------------------------------
# Random agent (for baselines / exploration)
# ---------------------------------------------------------------------------

class RandomContainerAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(
        self,
        state: dict,
        objective: str,
        available_actions: list[dict],
    ) -> dict:
        if not available_actions:
            return {"endpoint": "/forge/state", "payload": {}}

        action = self._rng.choice(available_actions)
        endpoint = action["endpoint"]
        payload: dict = {}
        props = action.get("request_schema", {}).get("properties", {})
        for name, schema in props.items():
            t = schema.get("type", "string")
            if t == "integer":
                payload[name] = self._rng.randint(1, 100)
            elif t == "number":
                payload[name] = round(self._rng.uniform(0, 100), 2)
            elif t == "boolean":
                payload[name] = self._rng.choice([True, False])
            else:
                payload[name] = f"test_{name}_{self._rng.randint(1, 10)}"
        return {"endpoint": endpoint, "payload": payload}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_container_agent(agent_id: str, seed: int | None = None) -> ContainerAgentBase:
    """Create a container agent from an agent_id string.

    Supported IDs:
      "random"              — RandomContainerAgent
      "llm"                 — LLMContainerAgent with default model (Haiku)
      "llm:<model_id>"      — LLMContainerAgent with the specified model
    """
    if agent_id == "random":
        return RandomContainerAgent(seed=seed)
    if agent_id == "llm" or agent_id.startswith("llm:"):
        model = agent_id[4:] if agent_id.startswith("llm:") else "claude-haiku-4-5-20251001"
        return LLMContainerAgent(
            get_client(max_tokens=envgen_config().action_llm_tokens, model=model)
        )
    raise AgentError(f"Unknown container agent id: {agent_id!r}. Use 'random', 'llm', or 'llm:<model>'.")
