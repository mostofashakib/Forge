from __future__ import annotations


class ActionValidator:
    def __init__(self, valid_action_types: set[str]) -> None:
        self._valid_types = valid_action_types

    def validate(self, action: object) -> dict | None:
        if not isinstance(action, dict):
            return {
                "error": "INVALID_ACTION",
                "code": "INVALID_FORMAT",
                "detail": f"Action must be a dict, got {type(action).__name__}",
            }
        if "type" not in action:
            return {
                "error": "INVALID_ACTION",
                "code": "MISSING_TYPE",
                "detail": "Action must have a 'type' field",
            }
        if action["type"] not in self._valid_types:
            return {
                "error": "INVALID_ACTION",
                "code": "UNKNOWN_TYPE",
                "detail": (
                    f"Unknown action type: '{action['type']}'. "
                    f"Valid types: {sorted(self._valid_types)}"
                ),
            }
        return None
