from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from typing import Any


class UnsafeExpressionError(ValueError):
    pass


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def evaluate_expression(expression: str, context: Mapping[str, Any]) -> Any:
    """Evaluate the small expression language used by policies and verifiers.

    The language permits values, indexing, safe field access, comparisons,
    boolean/arithmetic operators, ``len(value)``, and mapping ``.get()``.
    It deliberately has no imports, arbitrary calls, comprehensions, lambdas,
    assignments, or access to private attributes.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(str(exc)) from exc
    return _evaluate(tree.body, context)


def _evaluate(node: ast.AST, context: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise UnsafeExpressionError(f"Unknown name: {node.id}")
        return context[node.id]
    if isinstance(node, ast.List):
        return [_evaluate(item, context) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(item, context) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _evaluate(key, context): _evaluate(value, context)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Subscript):
        return _evaluate(node.value, context)[_evaluate(node.slice, context)]
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise UnsafeExpressionError("Private attribute access is not allowed")
        value = _evaluate(node.value, context)
        if isinstance(value, Mapping):
            if node.attr not in value:
                raise UnsafeExpressionError(f"Unknown field: {node.attr}")
            return value[node.attr]
        return getattr(value, node.attr)
    if isinstance(node, ast.BoolOp):
        values = node.values
        if isinstance(node.op, ast.And):
            result = _evaluate(values[0], context)
            for item in values[1:]:
                if not result:
                    return result
                result = _evaluate(item, context)
            return result
        if isinstance(node.op, ast.Or):
            result = _evaluate(values[0], context)
            for item in values[1:]:
                if result:
                    return result
                result = _evaluate(item, context)
            return result
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, context)
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise UnsafeExpressionError(f"Unsupported operator: {type(node.op).__name__}")
        return operation(_evaluate(node.left, context), _evaluate(node.right, context))
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, context)
        for op_node, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate(comparator, context)
            operation = _COMPARISON_OPERATORS.get(type(op_node))
            if operation is None:
                raise UnsafeExpressionError(
                    f"Unsupported comparison: {type(op_node).__name__}"
                )
            if not operation(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        args = [_evaluate(arg, context) for arg in node.args]
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are not allowed")
        if isinstance(node.func, ast.Name) and node.func.id == "len" and len(args) == 1:
            return len(args[0])
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            owner = _evaluate(node.func.value, context)
            if isinstance(owner, Mapping) and 1 <= len(args) <= 2:
                return owner.get(*args)
        raise UnsafeExpressionError("Function call is not allowed")
    if isinstance(node, ast.IfExp):
        branch = node.body if _evaluate(node.test, context) else node.orelse
        return _evaluate(branch, context)
    raise UnsafeExpressionError(f"Unsupported expression: {type(node).__name__}")
