"""Output schemas, generated from the tool definitions.

Every model call carries a schema, so what comes back is constrained at decode
time rather than parsed hopefully. What that schema *is* differs by service, and
the difference is not cosmetic:

* Ollama constrains the whole message with ``format``, and ``format`` and
  ``tools`` are mutually exclusive -- send both and tool calls silently stop
  coming back. So the tool surface is expressed *as* the schema: one action
  object whose ``tool`` is an enum over the real tool names and whose
  ``arguments`` are that tool's own input schema, selected by a discriminated
  union.
* OpenAI-compatible endpoints constrain function arguments with ``strict: true``
  on the function definition, which is the same guarantee reached the other way
  round -- but strict mode has its own rules about what a schema may look like,
  so ours are normalized to meet them.
* Anthropic validates ``input_schema`` server-side and returns arguments already
  decoded.

A schema pins the shape and says nothing about meaning, so :func:`render_catalogue`
puts the names, arguments and descriptions in the prompt. Without it a model in
schema mode picks a plausible-looking tool and fills it with empty strings --
structurally perfect and completely wrong.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

#: The key the loop reads a final answer out of, and the one it reads actions
#: from. Named here because the schema, the prompt and the decoder all have to
#: agree and there is no reason for three spellings.
THOUGHT_KEY = "thought"
ACTIONS_KEY = "actions"
ANSWER_KEY = "answer"


def build_action_schema(tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One turn of output, as a JSON schema.

    ``actions`` is an array so a model can act more than once in a turn, and
    empty so it can stop: no actions plus an ``answer`` is how an episode ends.
    That maps onto the same ``Completion`` a native tool-calling provider
    returns, which is why the loop never learns which mode it is in.
    """
    variants = [
        {
            "type": "object",
            "properties": {
                "tool": {"const": tool["name"]},
                "arguments": _as_object_schema(tool.get("input_schema")),
            },
            "required": ["tool", "arguments"],
            "additionalProperties": False,
        }
        for tool in tools
    ]
    if not variants:
        raise ValueError("An action schema needs at least one tool.")
    item: dict[str, Any] = variants[0] if len(variants) == 1 else {"oneOf": variants}
    return {
        "type": "object",
        "properties": {
            THOUGHT_KEY: {"type": "string"},
            ACTIONS_KEY: {"type": "array", "items": item},
            ANSWER_KEY: {"type": "string"},
        },
        "required": [THOUGHT_KEY, ACTIONS_KEY, ANSWER_KEY],
        "additionalProperties": False,
    }


def render_catalogue(tools: Sequence[Mapping[str, Any]]) -> str:
    """The tools as prose, for the prompt.

    The schema makes the wrong shape unrepresentable; this is what makes the
    right tool findable. A `?` marks an optional argument and a bracketed list
    is the accepted values.
    """
    lines = ["Available actions:"]
    for tool in tools:
        schema = _as_object_schema(tool.get("input_schema"))
        properties: dict[str, Any] = schema.get("properties", {})
        required = set(schema.get("required", []))
        rendered = ", ".join(
            f"{name}{'' if name in required else '?'}: {_render_type(spec)}"
            for name, spec in properties.items()
        )
        lines.append(f"- {tool['name']}({rendered or ''})")
        description = str(tool.get("description", "")).strip()
        if description:
            lines.append(f"    {description}")
    return "\n".join(lines)


def _render_type(spec: Mapping[str, Any]) -> str:
    if "enum" in spec:
        return "|".join(str(value) for value in spec["enum"])
    kind = spec.get("type", "any")
    if kind == "array":
        return f"[{_render_type(spec.get('items') or {})}]"
    return str(kind)


# --------------------------------------------------------------------------
# OpenAI strict mode
# --------------------------------------------------------------------------


def strict_parameters(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    """Rewrite an input schema to what ``strict: true`` will accept.

    Strict mode requires every declared property to be listed in ``required``
    and every object to close itself. Genuinely optional arguments -- which is
    most of this tool surface -- become nullable instead, so the model still has
    a way to decline one. :func:`drop_nulls` takes those back out before the
    call reaches the world, which would refuse ``{"title": null}`` as a request
    to blank the title.
    """
    schema = _as_object_schema(schema)
    properties: dict[str, Any] = {}
    required = set(schema.get("required", []))
    for name, spec in (schema.get("properties") or {}).items():
        properties[name] = spec if name in required else _nullable(spec)
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _nullable(spec: Mapping[str, Any]) -> dict[str, Any]:
    relaxed = dict(spec)
    kind = relaxed.get("type", "string")
    relaxed["type"] = [*kind, "null"] if isinstance(kind, list) else [kind, "null"]
    if "enum" in relaxed and None not in relaxed["enum"]:
        relaxed["enum"] = [*relaxed["enum"], None]
    return relaxed


def drop_nulls(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the nulls strict mode made the model write.

    An omitted argument and an argument explicitly set to null mean different
    things to the tracker, and only the first is what the model meant.
    """
    return {key: value for key, value in arguments.items() if value is not None}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class SchemaViolation(ValueError):
    """A payload that does not match the schema it was supposed to be made of."""


def validate(payload: Any, schema: Mapping[str, Any], path: str = "arguments") -> None:
    """Check ``payload`` against the subset of JSON Schema these tools use.

    Not a general validator, and not meant to be: it covers ``type``, ``enum``,
    ``const``, ``properties``, ``required``, ``additionalProperties``, ``items``
    and ``oneOf``, which is everything the generated schemas contain.

    The world validates too, and its answer is the authoritative one. This runs
    first so that a violation is a schema error the model can act on rather than
    a round trip, and so that the guarantee survives a provider added later
    whose constrained decoding is weaker than it claims -- the loop refuses to
    dispatch something the schema does not describe, whoever produced it.
    """
    if "oneOf" in schema:
        for variant in schema["oneOf"]:
            try:
                validate(payload, variant, path)
            except SchemaViolation:
                continue
            return
        raise SchemaViolation(f"{path}: matches none of the allowed shapes")

    if "const" in schema and payload != schema["const"]:
        raise SchemaViolation(f"{path}: expected {schema['const']!r}, got {payload!r}")
    if "enum" in schema and not _in_enum(payload, schema["enum"]):
        allowed = ", ".join(json.dumps(value) for value in schema["enum"])
        raise SchemaViolation(f"{path}: {payload!r} is not one of [{allowed}]")

    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else list(kinds)
        if not any(_is_kind(payload, kind) for kind in kinds):
            raise SchemaViolation(
                f"{path}: expected {'/'.join(kinds)}, got {type(payload).__name__}"
            )

    if isinstance(payload, dict) and "properties" in schema:
        properties: dict[str, Any] = schema["properties"]
        for name in schema.get("required", []):
            if name not in payload:
                raise SchemaViolation(f"{path}: missing required '{name}'")
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(payload) - set(properties))
            if unexpected:
                known = ", ".join(sorted(properties)) or "none"
                raise SchemaViolation(
                    f"{path}: unknown argument(s) {', '.join(unexpected)}; accepted: {known}"
                )
        for name, value in payload.items():
            if name in properties:
                validate(value, properties[name], f"{path}.{name}")

    if isinstance(payload, list) and "items" in schema:
        for index, value in enumerate(payload):
            validate(value, schema["items"], f"{path}[{index}]")


def _in_enum(value: Any, allowed: Sequence[Any]) -> bool:
    """Enum membership, case-insensitively for strings.

    The tracker normalizes status and priority on the way in and accepts
    `"pending"` for `PENDING`. Refusing that here would be this validator
    inventing a rule the world does not have, and a client that rejects calls
    the world would have honoured is worse than no client validation at all.
    """
    if value in allowed:
        return True
    if not isinstance(value, str):
        return False
    return value.upper() in {str(option).upper() for option in allowed if isinstance(option, str)}


_KINDS: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "array": list,
    "object": dict,
}


def _is_kind(value: Any, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    expected = _KINDS.get(kind)
    if expected is None:
        return True
    # In JSON, True is not a number and not a string, whatever Python thinks.
    if isinstance(value, bool) and expected is not dict:
        return False
    return isinstance(value, expected)


def _as_object_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    return dict(schema)
