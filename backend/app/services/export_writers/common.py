from __future__ import annotations

import json


def action_to_command(action_raw: str) -> str:
    try:
        action = json.loads(action_raw) if action_raw else {}
    except (json.JSONDecodeError, TypeError):
        return str(action_raw)
    return action.get("command") or action.get("cmd") or json.dumps(action)
