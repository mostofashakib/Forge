from __future__ import annotations
import json

from pydantic import BaseModel

from forge.extraction.llm_client import LLMClient, get_client
from forge.envgen.config import envgen_config


class _ScoreSchema(BaseModel):
    score: float
    reasoning: str


_SCORER_SYSTEM = (
    "You are evaluating how well a web application's current state achieves a stated objective.\n"
    "Score from 0.0 (no progress at all) to 1.0 (objective fully and completely achieved).\n"
    "Scoring guide:\n"
    "  0.0-0.1 — no meaningful progress, wrong direction\n"
    "  0.2-0.4 — some minor progress but far from done\n"
    "  0.5-0.7 — meaningful partial progress toward the objective\n"
    "  0.8-0.9 — mostly achieved, minor gaps remain\n"
    "  1.0     — objective fully achieved\n"
    "Be concise. Call the extract tool with your numeric score and a one-sentence reasoning."
)


class ObjectivePrompts:
    SYSTEM = _SCORER_SYSTEM


class ObjectiveScorer:
    """LLM-based scorer that evaluates how well a state achieves an objective."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_client(max_tokens=envgen_config().cli_llm_tokens)

    def score(
        self,
        state: dict,
        objective: str,
        *,
        derived_diff: dict | None = None,
        action_taken: dict | None = None,
    ) -> float:
        """Return 0.0–1.0 representing how well state achieves objective.

        Falls back to 0.5 on any LLM/network error.
        """
        try:
            state_text = json.dumps(state, indent=2)
            if len(state_text) > 3000:
                state_text = state_text[:3000] + "\n... (truncated)"
            user = f"Objective: {objective}\n\nCurrent application state:\n{state_text}"
            if derived_diff:
                user += f"\n\nDerived field changes (populated by this action):\n{json.dumps(derived_diff, indent=2)}"
            if action_taken:
                user += f"\n\nAction taken: {json.dumps(action_taken)}"
            result = self._client.extract(
                system=ObjectivePrompts.SYSTEM, user=user, schema=_ScoreSchema
            )
            return max(0.0, min(1.0, float(result.score)))
        except Exception:
            return 0.5

    def score_with_image(self, screenshot_b64: str, url: str, objective: str) -> float:
        """Score a browser state using a screenshot. Falls back to 0.5 on error."""
        try:
            client = get_client(max_tokens=envgen_config().cli_llm_tokens)
            user = f"Objective: {objective}\n\nCurrent URL: {url}\n\nSee the screenshot for the current browser state."
            result = client.extract_with_image(
                system=ObjectivePrompts.SYSTEM, user=user,
                image_b64=screenshot_b64, schema=_ScoreSchema,
            )
            return max(0.0, min(1.0, float(result.score)))
        except Exception:
            return 0.5
