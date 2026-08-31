"""Reserved episode-control actions understood by every controller."""
from __future__ import annotations

from typing import Any

SUBMIT_ACTION = "submit"
SUBMIT_ENDPOINT = "__forge_submit__"
SUBMIT_COMMAND = "FORGE_SUBMIT"


def is_submit_action(action: Any) -> bool:
    """Return whether an agent action asks Forge to grade and end the episode."""
    if isinstance(action, str):
        return action.strip() == SUBMIT_COMMAND
    if not isinstance(action, dict):
        return False
    return (
        action.get("type") == SUBMIT_ACTION
        or action.get("action_type") == SUBMIT_ACTION
        or action.get("endpoint") == SUBMIT_ENDPOINT
    )
