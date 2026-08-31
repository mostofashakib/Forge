"""Arity checking for the subclass methods engines call with a fixed shape.

`@abstractmethod` guarantees only that a name exists on the subclass, never
that its signature matches — a generated `class Foo(TransitionHandler): def
apply(self, state): ...` satisfies it and then fails mid-episode, the exact
defect the contracts exist to eliminate. `__init_subclass__` on the affected
ABCs calls `check_subclass_arity` so a mismatched shape is rejected when the
subclass is *defined*, not when it is first called.

This module intentionally duplicates the spirit of
`forge/runtime/_signature.py:require_arity` (binding placeholder arguments
rather than counting parameters, so `*args`, defaults, and keyword-only
extras with defaults all keep working). `forge/contracts/` may only import
`forge/schema/`, the standard library, and pydantic at runtime, so the two
copies cannot share code across that boundary — see
`tests/contracts/test_import_direction.py`.
"""
from __future__ import annotations

import inspect


def check_subclass_arity(
    cls: type, method_name: str, params: tuple[str, ...]
) -> None:
    """Raise TypeError if `cls` defines `method_name` with the wrong arity.

    Skipped entirely when `cls` does not itself define `method_name` (an
    intermediate ABC that stays abstract must remain definable).
    """
    method = cls.__dict__.get(method_name)
    if method is None:
        return
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        # No introspectable signature (e.g. some C-implemented callables) —
        # nothing to check, so accept rather than reject.
        return
    try:
        # `self` plus one placeholder per required param.
        signature.bind(*(None,) * (len(params) + 1))
    except TypeError as exc:
        raise TypeError(
            f"{cls.__name__}.{method_name} must accept (self, "
            f"{', '.join(params)}), but its signature is "
            f"{method_name}{signature}: {exc}"
        ) from exc
