from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from forge.runtime.reward import FunctionRubric
from forge.runtime.transition import FunctionTransitionHandler
from forge.runtime.verifier import FunctionVerifier

# The three contract-backed hooks store a wrapped contract instance; the two
# without a contract (observation transforms, policy rules) stay plain
# functions, because their callers invoke them directly.
_registry: dict[str, dict[str, Any]] = {
    "transitions": {},
    "verifiers": {},
    "rewards": {},
    "observation_transforms": {},
    "policy_rules": {},
}


def clear_registry() -> None:
    for v in _registry.values():
        v.clear()


def get_registry() -> dict[str, dict[str, Any]]:
    return {k: dict(v) for k, v in _registry.items()}


def override_transition(action_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        # The registry stores the contract instance; the author keeps their
        # plain function, so the decorator stays transparent at the call site.
        _registry["transitions"][action_name] = FunctionTransitionHandler(fn)
        return fn
    return decorator


def verifier(task_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _registry["verifiers"][task_name] = FunctionVerifier(fn)
        return fn
    return decorator


def reward(task_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _registry["rewards"][task_name] = FunctionRubric(fn)
        return fn
    return decorator


def observation_transform(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _registry["observation_transforms"][name] = fn
        return fn
    return decorator


def policy_rule(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _registry["policy_rules"][name] = fn
        return fn
    return decorator


def import_custom_file(path: Path, module_prefix: str = "_forge_custom") -> None:
    """Import a Python file by path into sys.modules."""
    module_name = f"{module_prefix}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
