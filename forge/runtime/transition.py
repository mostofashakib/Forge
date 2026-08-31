from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable

from forge.contracts import Action
from forge.contracts.backend import TransitionHandler
from forge.runtime._signature import require_arity
from forge.runtime.context import RuntimeContext
from forge.runtime.snapshot import InvalidActionError


@dataclass
class TransitionResult:
    state: dict
    events: list[dict] = field(default_factory=list)


class FunctionTransitionHandler(TransitionHandler):
    """Adapts a plain `(state, action, ctx) -> TransitionResult` function.

    Used by the customization hooks so a decorated function stays as easy to
    write as it was before the contract existed. The wrapped function still
    receives the action as a plain dict, which is what every hand-written and
    generated handler expects.
    """

    def __init__(self, fn: Callable) -> None:
        # The registry can only check kind; arity is knowable here, so a
        # mis-declared handler fails at decoration/build time, not mid-episode.
        require_arity(fn, "FunctionTransitionHandler", ("state", "action", "ctx"))
        self._fn = fn
        # Carry the author's name/doc onto the adapter so tracebacks and
        # registry introspection still identify the original function.
        functools.update_wrapper(self, fn, updated=())

    @property
    def fn(self) -> Callable:
        """The plain function this adapter wraps."""
        return self._fn

    def apply(
        self, state: dict, action: Action, ctx: RuntimeContext
    ) -> TransitionResult:
        return self._fn(state, action.to_dict(), ctx)


class TransitionEngine:
    def __init__(self) -> None:
        self._handlers: dict[str, TransitionHandler] = {}

    def register(self, action_type: str, handler: TransitionHandler) -> None:
        if not isinstance(handler, TransitionHandler):
            raise TypeError(
                f"Handler for {action_type!r} must be a TransitionHandler, got "
                f"{type(handler).__name__}. Wrap a plain function in "
                f"FunctionTransitionHandler."
            )
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
        return handler.apply(state, Action.from_dict(action), ctx)
