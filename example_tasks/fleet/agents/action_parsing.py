"""Pure helpers for normalizing LLM tool actions."""

from __future__ import annotations

import re
from typing import Any


def normalize_ollama_action(action: dict[str, Any], available_tool_names: set[str]) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise RuntimeError(f"Ollama action must be a JSON object: {action!r}")

    if "final_answer" in action and "type" not in action:
        return {"type": "final_answer", "answer": action.get("final_answer", "")}
    if "answer" in action and "type" not in action and "tool_name" not in action:
        return {"type": "final_answer", "answer": action.get("answer", "")}
    if "tool_name" in action and "type" not in action:
        tool_input = action.get("input", action.get("input_payload", action.get("arguments", {})))
        return normalize_tool_call(str(action.get("tool_name", "")), tool_input, available_tool_names)
    if "function_name" in action and "type" not in action:
        tool_input = action.get("arguments", action.get("input", action.get("input_payload", {})))
        return normalize_tool_call(str(action.get("function_name", "")), tool_input, available_tool_names)

    action_type = str(action.get("type", "")).strip()
    if action_type == "final_answer":
        return {"type": "final_answer", "answer": action.get("answer", action.get("final_answer", ""))}
    if action_type in {"tool", "tool_call", "function_call"}:
        tool_name = str(action.get("tool_name", action.get("function_name", action.get("name", "")))).strip()
        tool_input = action.get("input", action.get("input_payload", action.get("arguments", {})))
        return normalize_tool_call(tool_name, tool_input, available_tool_names)
    raise RuntimeError(f"Unsupported action type from Ollama: {action_type}. Raw action: {action!r}")


def normalize_tool_call(tool_name: str, tool_input: Any, available_tool_names: set[str]) -> dict[str, Any]:
    tool_name = tool_name.strip()
    if tool_name not in available_tool_names:
        raise RuntimeError(f"Ollama requested unavailable tool: {tool_name}")
    if not isinstance(tool_input, dict):
        raise RuntimeError(f"Ollama tool input must be an object: {tool_input!r}")
    return {"type": "tool_call", "tool_name": tool_name, "input": tool_input}
