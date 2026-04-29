from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError


@dataclass
class TransitionResult:
    state: dict
    events: list[dict] = field(default_factory=list)


class TransitionEngine:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable] = {}

    def register(self, action_type: str, handler: Callable) -> None:
        self._handlers[action_type] = handler

    @property
    def action_types(self) -> set[str]:
        return set(self._handlers.keys())

    def apply(self, state: dict, action: dict, ctx: RuntimeContext) -> TransitionResult:
        handler = self._handlers.get(action.get("type", ""))
        if handler is None:
            raise InvalidActionError(
                f"Unknown action type: '{action.get('type')}'. Valid: {sorted(self._handlers)}",
                code="UNKNOWN_ACTION_TYPE",
            )
        return handler(state, action, ctx)
