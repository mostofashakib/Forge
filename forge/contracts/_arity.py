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

Two wrinkles beyond a plain function definition, found in fix round 1:

- The method may arrive from anywhere in the MRO, not just `cls.__dict__` —
  a mixin base can supply a concrete (non-abstract) implementation that
  `@abstractmethod` considers satisfied. Resolving through `getattr(cls, ...)`
  catches that; reading only `cls.__dict__` would miss it entirely.
- The method may be wrapped in `@classmethod` or `@staticmethod`. Both
  descriptors raise from `inspect.signature()` when handed the raw
  descriptor object, and both change how many parameters are implicit
  (`cls`, or none) versus explicit. Unwrapping via `.__func__` and adjusting
  the implicit-argument count keeps both introspectable and correctly
  counted, instead of accidentally falling into the "not introspectable,
  accept" fallback meant only for genuinely opaque C callables.
"""
from __future__ import annotations

import inspect


def check_subclass_arity(
    cls: type, method_name: str, params: tuple[str, ...]
) -> None:
    """Raise TypeError if `cls` provides `method_name` with the wrong arity.

    Resolved through the MRO (`getattr`), so a concrete implementation
    arriving from a mixin base is checked exactly like one defined directly
    on `cls`. Skipped when the method is missing entirely, or when the
    resolved attribute is still abstract — an intermediate ABC that doesn't
    touch this method, or that stays abstract on purpose, must remain
    definable.
    """
    resolved = getattr(cls, method_name, None)
    if resolved is None or getattr(resolved, "__isabstractmethod__", False):
        return

    # `getattr` above already binds `classmethod`/`staticmethod` descriptors
    # into a plain callable, which hides which kind they were. Fetch the raw,
    # undecorated descriptor (bypassing the descriptor protocol) to tell a
    # classmethod/staticmethod from a plain function, and to know how many
    # leading parameters are implicit rather than part of `params`.
    raw = inspect.getattr_static(cls, method_name)
    if isinstance(raw, staticmethod):
        func = raw.__func__
        implicit_count = 0
        implicit_label = ""
    elif isinstance(raw, classmethod):
        func = raw.__func__
        implicit_count = 1
        implicit_label = "cls, "
    else:
        func = raw
        implicit_count = 1
        implicit_label = "self, "

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        # No introspectable signature (e.g. some genuinely opaque
        # C-implemented callable) — nothing to check, so accept rather than
        # reject. classmethod/staticmethod never reach here: they were
        # unwrapped to their underlying function above.
        return
    try:
        signature.bind(*(None,) * (len(params) + implicit_count))
    except TypeError as exc:
        raise TypeError(
            f"{cls.__name__}.{method_name} must accept "
            f"({implicit_label}{', '.join(params)}), but its signature is "
            f"{method_name}{signature}: {exc}"
        ) from exc
