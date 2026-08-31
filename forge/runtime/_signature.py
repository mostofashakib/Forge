"""Arity checking for the plain callables the Function* adapters wrap.

`isinstance` proves an object's *kind*, not its *arity*: an adapter will happily
wrap a two-argument function and only fail when the engine calls it, mid-episode
— the exact defect the contracts exist to eliminate. The adapters are the one
place where a plain callable enters the typed world, so the shape check belongs
in their constructors.

Binding placeholder arguments to the real signature is used rather than counting
parameters, because `*args`, bound methods, `functools.partial`, and defaults are
all legitimate authoring shapes that a raw count would reject.
"""
from __future__ import annotations

import inspect
from typing import Callable


def require_arity(fn: Callable, adapter: str, params: tuple[str, ...]) -> None:
    """Raise TypeError unless `fn` accepts `len(params)` positional arguments."""
    if not callable(fn):
        raise TypeError(
            f"{adapter} expects a callable with the signature "
            f"({', '.join(params)}), got {type(fn).__name__}."
        )
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins and some C-implemented callables expose no signature. There
        # is nothing to check, so accept rather than reject a valid callable.
        return
    try:
        signature.bind(*(None,) * len(params))
    except TypeError as exc:
        name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
        raise TypeError(
            f"{adapter} expects a callable with the signature "
            f"({', '.join(params)}), but {name}{signature} cannot accept "
            f"{len(params)} positional arguments: {exc}"
        ) from exc
