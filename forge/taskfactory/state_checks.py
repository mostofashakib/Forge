"""Seed and check the state of in-process and app environments.

Collections are found by a dotted path and are either a list of row dicts or
a dict of row dicts keyed by id. Pure functions: no I/O, inputs untouched.
"""
from __future__ import annotations

import copy
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from forge.taskfactory.schemas import Check


@dataclass(frozen=True)
class CheckOutcome:
    passed: bool
    detail: str = ""


_COMPARE = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "contains": lambda left, right: right in left,
}
COMPARE_OPS: frozenset[str] = frozenset(_COMPARE)


def resolve_path(state: Any, path: str) -> Any:
    """Walk `a.b.0.c` through dicts and lists. Raises KeyError when absent."""
    node = state
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise KeyError(path)
    return node


def _rows(collection: Any) -> list[dict]:
    if isinstance(collection, list):
        return [row for row in collection if isinstance(row, dict)]
    if isinstance(collection, dict):
        return [row for row in collection.values() if isinstance(row, dict)]
    raise TypeError("not a collection")


def collection_rows(state: dict, path: str) -> list[dict]:
    """The rows at `path`. Raises KeyError or TypeError when it is not a collection."""
    return _rows(resolve_path(state, path))


def apply_seed(state: dict, records: dict[str, list[dict]]) -> dict:
    """Return `state` with each seed row merged into its collection by id."""
    seeded = copy.deepcopy(state)
    for path, rows in records.items():
        try:
            collection = resolve_path(seeded, path)
        except KeyError:
            raise ValueError(f"seed collection {path!r} does not exist in the state") from None
        for row in rows:
            _merge_row(collection, path, row)
    return seeded


def _merge_row(collection: Any, path: str, row: dict) -> None:
    row_id = row.get("id")
    if isinstance(collection, dict):
        if row_id is None:
            raise ValueError(f"rows seeded into {path!r} need an 'id'")
        collection.setdefault(str(row_id), {}).update(copy.deepcopy(row))
        return
    if not isinstance(collection, list):
        raise ValueError(f"seed collection {path!r} is not a list or dict of rows")
    existing = next(
        (item for item in collection if isinstance(item, dict) and row_id is not None and item.get("id") == row_id),
        None,
    )
    if existing is not None:
        existing.update(copy.deepcopy(row))
    else:
        collection.append(copy.deepcopy(row))


def _matches(row: dict, fields: dict | None) -> bool:
    return all(row.get(key) == value for key, value in (fields or {}).items())


def evaluate_state_check(check: Check, state: dict, called_tools: Sequence[str]) -> CheckOutcome:
    """Evaluate one state-family check. Never raises: a bad path is a failure."""
    kind = check.kind
    if kind == "actions_called":
        return _actions_called(check.tools or [], list(called_tools), check.ordered)
    if kind == "actions_not_called":
        ran = sorted(set(check.tools or []) & set(called_tools))
        return CheckOutcome(not ran, f"forbidden tools ran: {ran}" if ran else "")
    try:
        if kind == "value":
            actual = resolve_path(state, check.path or "")
            passed = _COMPARE[check.op or "=="](actual, check.value)
            return CheckOutcome(passed, "" if passed else f"{check.path} is {actual!r}, wanted {check.op} {check.value!r}")
        rows = collection_rows(state, check.collection or "")
    except (KeyError, TypeError) as exc:
        return CheckOutcome(False, f"cannot read {check.path or check.collection!r}: {exc}")
    matching = [row for row in rows if _matches(row, check.match)]
    if kind == "record_exists":
        if any(_matches(row, check.expect) for row in matching):
            return CheckOutcome(True)
        return CheckOutcome(False, f"no row in {check.collection} matches {check.match} with {check.expect}")
    if kind == "record_absent":
        return CheckOutcome(not matching, f"{len(matching)} rows in {check.collection} match {check.match}" if matching else "")
    if kind == "record_count":
        passed = _COMPARE[check.op or "=="](len(matching), check.value)
        return CheckOutcome(passed, "" if passed else f"{len(matching)} rows match {check.match}, wanted {check.op} {check.value}")
    return CheckOutcome(False, f"{kind!r} is not a state check")


def _actions_called(wanted: list[str], called: list[str], ordered: bool) -> CheckOutcome:
    if not ordered:
        missing = [tool for tool in wanted if tool not in called]
        return CheckOutcome(not missing, f"never called: {missing}" if missing else "")
    position = 0
    for tool in wanted:
        try:
            position = called.index(tool, position) + 1
        except ValueError:
            return CheckOutcome(False, f"{wanted} were not called in that order")
    return CheckOutcome(True)
