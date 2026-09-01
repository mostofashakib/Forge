"""The boundary between what a persona driver proposes and what the world runs.

A driver may call a model, and a model will occasionally invent an action, name
a tool it was never given, or hand a persona a capability the environment
author never meant them to have. None of that is allowed to reach state. Every
proposal passes through `ActionGuard`, which answers one question — is this
persona permitted to do this, right now — and a rejection is recorded on the
turn rather than raised, so a misbehaving driver degrades a single persona turn
instead of failing the episode.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from forge.contracts.persona import PersonaSpec
from forge.contracts.types import Action, ToolSpec


class GuardDecision:
    """Whether one proposed action may run, and why not when it may not."""

    __slots__ = ("allowed", "reason")

    def __init__(self, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason

    def __bool__(self) -> bool:
        return self.allowed

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GuardDecision(allowed={self.allowed!r}, reason={self.reason!r})"


ALLOWED = GuardDecision(True)


class ActionGuard:
    """Enforces a persona's declared action space against what a driver picked.

    Four checks, in the order a violation is most likely:

    1. The persona declared *some* action space at all.
    2. The action type is in that space.
    3. The environment actually implements that action type — a persona cannot
       be granted an action the environment does not have, which catches the
       common configuration drift of renaming an action and forgetting the
       persona roster.
    4. Required parameters of the action's schema are present.

    `environment_actions` is optional. Left unset, check 3 is skipped, so the
    guard is usable in a context that does not know the environment's surface
    (the configuration UI previewing a roster, for instance) without silently
    weakening checks 1, 2, and 4.
    """

    def __init__(
        self,
        environment_actions: Iterable[str] | None = None,
        tool_specs: Sequence[ToolSpec] | None = None,
    ) -> None:
        self._environment_actions = (
            frozenset(environment_actions) if environment_actions is not None else None
        )
        self._tool_specs = {spec.name: spec for spec in (tool_specs or ())}

    def check(self, spec: PersonaSpec, action: Action) -> GuardDecision:
        allowed = spec.behavior.allowed_actions
        if not allowed:
            return GuardDecision(
                False,
                f"persona {spec.profile.id!r} has no allowed_actions configured, "
                "so it may not act",
            )

        if action.type not in allowed:
            return GuardDecision(
                False,
                f"persona {spec.profile.id!r} proposed {action.type!r}, which is "
                f"outside its action space {sorted(allowed)}",
            )

        if (
            self._environment_actions is not None
            and action.type not in self._environment_actions
        ):
            return GuardDecision(
                False,
                f"action {action.type!r} is in persona {spec.profile.id!r}'s "
                "action space but the environment does not implement it",
            )

        missing = self._missing_params(action)
        if missing:
            return GuardDecision(
                False,
                f"persona {spec.profile.id!r} proposed {action.type!r} without "
                f"required parameters: {', '.join(missing)}",
            )

        return ALLOWED

    def action_space(self, spec: PersonaSpec) -> list[ToolSpec]:
        """The persona's permitted actions as tool schemas, in a stable order.

        Actions the environment does not implement are dropped rather than
        shown: a driver that never sees a stale action type cannot propose one.
        The guard still rejects it if some other driver does.
        """
        names = sorted(set(spec.behavior.allowed_actions))
        if self._environment_actions is not None:
            names = [name for name in names if name in self._environment_actions]
        return [self._tool_specs.get(name, ToolSpec(name=name)) for name in names]

    def _missing_params(self, action: Action) -> list[str]:
        schema = self._tool_specs.get(action.type)
        if schema is None:
            return []
        return [
            param.name
            for param in schema.params
            if param.required and param.name not in action.params
        ]
