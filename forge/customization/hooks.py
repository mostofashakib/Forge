from __future__ import annotations
from typing import Callable

_registry: dict[str, dict[str, Callable]] = {
    "transitions": {},
    "verifiers": {},
    "rewards": {},
    "observation_transforms": {},
    "policy_rules": {},
}


def clear_registry() -> None:
    for v in _registry.values():
        v.clear()


def get_registry() -> dict[str, dict[str, Callable]]:
    return {k: dict(v) for k, v in _registry.items()}


def override_transition(action_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _registry["transitions"][action_name] = fn
        return fn
    return decorator


def verifier(task_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _registry["verifiers"][task_name] = fn
        return fn
    return decorator


def reward(task_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _registry["rewards"][task_name] = fn
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
