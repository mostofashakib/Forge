from __future__ import annotations
import json
import random

from pydantic import BaseModel

from forge.extraction.llm_client import LLMClient, get_client
from forge.runtime.errors import AgentError
from forge.envgen.config import envgen_config


_BROWSER_SYSTEM = (
    "You are an autonomous browser agent. You see a screenshot of a web browser "
    "and must choose the single best action to achieve the objective.\n"
    "\n"
    "Available action types:\n"
    '  click    — {"action_type": "click", "x": <int>, "y": <int>}\n'
    '  type     — {"action_type": "type", "text": "<string>"}\n'
    '  press    — {"action_type": "press", "key": "<Enter|Tab|Backspace|Escape|...>"}\n'
    '  navigate — {"action_type": "navigate", "url": "<full url>"}\n'
    '  scroll   — {"action_type": "scroll", "delta_x": <int>, "delta_y": <int>}\n'
    '  noop     — {"action_type": "noop"}\n'
    "\n"
    "Coordinates (x, y) are pixels from the top-left corner of the viewport.\n"
    "If the objective is already achieved, use noop.\n"
    "Include your one-sentence reasoning."
)


class BrowserAgentPrompts:
    SYSTEM = _BROWSER_SYSTEM


class _BrowserActionSchema(BaseModel):
    action_type: str
    x: int | None = None
    y: int | None = None
    text: str | None = None
    key: str | None = None
    url: str | None = None
    delta_x: int | None = None
    delta_y: int | None = None
    reasoning: str


class LLMBrowserAgent:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client(max_tokens=envgen_config().action_llm_tokens)

    def act(
        self,
        screenshot_b64: str,
        objective: str,
        action_history: list[dict],
    ) -> dict:
        user = (
            f"Objective: {objective}\n\n"
            f"Recent actions:\n{json.dumps(action_history[-3:], indent=2)}\n\n"
            "The screenshot shows the current browser state."
        )
        result = self._client.extract_with_image(
            system=BrowserAgentPrompts.SYSTEM,
            user=user,
            image_b64=screenshot_b64,
            schema=_BrowserActionSchema,
        )
        action: dict = {"action_type": result.action_type, "reasoning": result.reasoning}
        if result.x is not None:       action["x"] = result.x
        if result.y is not None:       action["y"] = result.y
        if result.text is not None:    action["text"] = result.text
        if result.key is not None:     action["key"] = result.key
        if result.url is not None:     action["url"] = result.url
        if result.delta_x is not None: action["delta_x"] = result.delta_x
        if result.delta_y is not None: action["delta_y"] = result.delta_y
        return action


class RandomBrowserAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(
        self,
        screenshot_b64: str,
        objective: str,
        action_history: list[dict],
    ) -> dict:
        choices = [
            {"action_type": "click",  "x": self._rng.randint(50, 900), "y": self._rng.randint(50, 700)},
            {"action_type": "scroll", "delta_x": 0, "delta_y": self._rng.choice([300, -300])},
            {"action_type": "press",  "key": "Return"},
            {"action_type": "noop"},
        ]
        return {**self._rng.choice(choices), "reasoning": "random action"}


def make_browser_agent(agent_id: str, seed: int | None = None):
    if agent_id == "random":
        return RandomBrowserAgent(seed=seed)
    if agent_id == "llm" or agent_id.startswith("llm:"):
        model = agent_id[4:] if agent_id.startswith("llm:") else None
        return LLMBrowserAgent(
            get_client(max_tokens=envgen_config().action_llm_tokens, model=model)
        )
    raise AgentError(f"Unknown browser agent id: {agent_id!r}. Use 'random', 'llm', or 'llm:<model>'.")
