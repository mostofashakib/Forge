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

Three more holes in that same "not introspectable, accept" fallback, found
in fix round 2 — all three let a defect of the exact class this module
exists to catch pass silently, or rejected a shape that was actually fine:

- `functools.partialmethod` also raises TypeError from `inspect.signature()`
  on the raw descriptor, so a *wrong-arity* partialmethod fell into the
  same accept-by-default fallback as classmethod/staticmethod did before
  fix round 1. It is unwrapped explicitly: the descriptor's own bound
  `args`/`keywords` are spliced into a simulated call alongside `self` and
  placeholder `params`, rather than treated as opaque.
- A `property` shadowing a contract method also raises TypeError from
  `inspect.signature()` (a property is not callable at all), so it too fell
  into the same fallback. Unlike partialmethod there is no arity to check —
  a property is read as an attribute, never called — so it is rejected
  outright rather than arity-checked.
- A correctly-shaped **callable instance** (`apply = SomeCallable()`) was
  *over-rejected*: the code assumed every non-static/classmethod attribute
  is a plain function reached through the descriptor protocol and so has an
  implicit `self`, but an arbitrary object with only `__call__` is not a
  descriptor — accessing it never binds `self`. `inspect.isfunction()` now
  distinguishes the two cases explicitly instead of assuming the common one.
"""
from __future__ import annotations

import functools
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

    # A `property` can never satisfy a method contract: it is read as a
    # plain attribute access (`instance.apply`, no parentheses) and whatever
    # its getter returns is not called with `params` at all. It defines
    # cleanly and then fails at call time with an unrelated "not callable"
    # error, which is exactly the kind of "satisfies the shape, fails
    # mid-episode" defect this module exists to catch before that point.
    # Reject it outright rather than trying to arity-check the getter.
    if isinstance(raw, property):
        raise TypeError(
            f"{cls.__name__}.{method_name} is a property; a property is "
            f"read as an attribute, not called, so it cannot satisfy the "
            f"method contract {method_name}(self, {', '.join(params)})"
        )

    # `functools.partialmethod` is a descriptor but not a plain function, so
    # `inspect.signature()` raises TypeError directly on the raw object
    # (it "is not a callable object") rather than describing its shape.
    # Falling into the generic "not introspectable, accept" fallback below
    # would silently accept a wrong-arity partialmethod, exactly the defect
    # this module exists to catch. Unwrap to the underlying function and
    # simulate the real call: the descriptor supplies `self` first, then
    # the partialmethod's own bound `args`/`keywords`, then whatever the
    # engine passes positionally for `params`.
    if isinstance(raw, functools.partialmethod):
        func = raw.func
        try:
            func_signature = inspect.signature(func)
        except (TypeError, ValueError):
            return
        try:
            func_signature.bind(
                None, *raw.args, *(None,) * len(params), **raw.keywords
            )
        except TypeError as exc:
            raise TypeError(
                f"{cls.__name__}.{method_name} (a partialmethod wrapping "
                f"{getattr(func, '__qualname__', func)}) must accept "
                f"(self, {', '.join(params)}) once its bound arguments are "
                f"applied, but its signature is {method_name}{func_signature}: "
                f"{exc}"
            ) from exc
        return

    if isinstance(raw, staticmethod):
        func = raw.__func__
        implicit_count = 0
        implicit_label = ""
    elif isinstance(raw, classmethod):
        func = raw.__func__
        implicit_count = 1
        implicit_label = "cls, "
    elif inspect.isfunction(raw):
        func = raw
        implicit_count = 1
        implicit_label = "self, "
    else:
        # Not a plain function descriptor — e.g. a callable instance
        # (`apply = SomeCallable()`) assigned as the class attribute.
        # Plain functions implement the descriptor protocol (`__get__`),
        # which is what turns `instance.apply` into a bound method with an
        # implicit `self`; an arbitrary object with only `__call__` does
        # not implement that protocol, so accessing `instance.apply` returns
        # the object itself and calling it never passes `self` at all.
        # `inspect.signature()` on the object already reflects exactly the
        # arguments it will be called with (it introspects `__call__` and
        # drops that method's own `self`), so no implicit count is added.
        func = raw
        implicit_count = 0
        implicit_label = ""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        # No introspectable signature (e.g. some genuinely opaque
        # C-implemented callable) — nothing to check, so accept rather than
        # reject. classmethod/staticmethod/partialmethod never reach here:
        # they were unwrapped to their underlying function above.
        return
    try:
        signature.bind(*(None,) * (len(params) + implicit_count))
    except TypeError as exc:
        raise TypeError(
            f"{cls.__name__}.{method_name} must accept "
            f"({implicit_label}{', '.join(params)}), but its signature is "
            f"{method_name}{signature}: {exc}"
        ) from exc
