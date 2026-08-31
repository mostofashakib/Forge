"""In-container agents for the SQLite Slack service.

These drive the service through :mod:`fleet.environments.slack.sqlite_service`
and write ATIF trajectories. The service module stays agent-free; its CLI
imports this module lazily for the ``run-reference-agent`` and
``run-ollama-agent`` commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fleet.agents.model_adapters import create_adapter
from fleet.core.atif import atif_trajectory_to_dict
from fleet.environments.slack.schema import available_tools
from fleet.environments.slack.sqlite_service import (
    execute_tool,
    export_state,
    get_channel_messages,
    search_messages,
    teardown_database,
)


def first_channel_reference(text: str) -> str:
    for token in text.split():
        cleaned = token.strip(".,;:!?()[]{}")
        if cleaned.startswith("#") and len(cleaned) > 1:
            return cleaned
    return ""


def build_trajectory(db_path: Path, instruction: str, search_result: dict[str, Any], channel_result: dict[str, Any], answer: str) -> dict[str, Any]:
    trajectory_id = "slack-incident-reference-seed-1"
    return atif_trajectory_to_dict(
        trajectory_id=trajectory_id,
        session_id=trajectory_id,
        agent_name="slack-installed-agent",
        agent_version="0.1.0",
        model_name=None,
        steps=[
            {"step_id": 1, "source": "user", "message": instruction},
            {
                "step_id": 2,
                "source": "agent",
                "message": "I need to search Alice's incident messages.",
                "tool_calls": [
                    {
                        "tool_call_id": "call_1",
                        "function_name": "search_messages",
                        "arguments": {"query": "alice incidents"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "call_1", "content": json.dumps(search_result, sort_keys=True)}]},
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "I need the #incidents channel history to confirm recency.",
                "tool_calls": [
                    {
                        "tool_call_id": "call_2",
                        "function_name": "get_channel_messages",
                        "arguments": {"channel_id": "C003"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "call_2", "content": json.dumps(channel_result, sort_keys=True)}]},
            },
            {"step_id": 4, "source": "agent", "message": answer},
        ],
        final_metrics={"total_steps": 4, "total_prompt_tokens": 0, "total_completion_tokens": 0, "total_cached_tokens": 0, "total_cost_usd": 0.0},
        extra={
            "final_answer": answer,
            "expected_answer": "#platform-outages",
            "final_state_snapshot": export_state(db_path),
        },
    )


def ollama_next_action(
    *,
    model: str,
    host: str,
    instruction: str,
    state: dict[str, Any],
    history: list[dict[str, Any]],
    timeout_sec: float,
) -> dict[str, Any]:
    prompt = "\n".join(
        [
            "You are an agent interacting with a deterministic Slack environment.",
            "Return exactly one JSON object and no markdown.",
            "To call a tool, return: {\"thought\":\"...\",\"tool_name\":\"...\",\"input_payload\":{...}}",
            "When done, return: {\"thought\":\"...\",\"final_answer\":\"...\"}",
            "Use only the listed tools. Do not invent IDs; read them from state or observations.",
            "Do not answer directly from state. You must call tools before returning final_answer.",
            "The final answer must be the referenced channel name only, for example #channel-name.",
            f"Instruction: {instruction}",
            f"Available tools: {json.dumps(available_tools(), sort_keys=True)}",
            f"Current state: {json.dumps(state, sort_keys=True)}",
            f"History: {json.dumps(history, sort_keys=True)}",
        ]
    )
    adapter = create_adapter(model, host.rstrip("/"), stream_stall_timeout_sec=timeout_sec)
    raw_response = adapter.generate(prompt, model)
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned non-JSON response: {raw_response!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Ollama returned JSON that is not an object: {parsed!r}")
    return parsed


def build_trajectory_from_history(
    *,
    db_path: Path,
    instruction: str,
    model: str,
    history: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [{"step_id": 1, "source": "user", "message": instruction}]
    for index, item in enumerate(history, start=1):
        call_id = f"call_{index}"
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": "agent",
                "model_name": model,
                "message": item["thought"],
                "reasoning_content": item["thought"],
                "tool_calls": [
                    {
                        "tool_call_id": call_id,
                        "function_name": item["tool_name"],
                        "arguments": item["input_payload"],
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": call_id,
                            "content": json.dumps(item["observation"], sort_keys=True),
                        }
                    ]
                },
                "llm_call_count": 1,
            }
        )
    steps.append(
        {
            "step_id": len(steps) + 1,
            "source": "agent",
            "model_name": model,
            "message": answer,
            "llm_call_count": 1,
        }
    )
    trajectory_id = "slack-incident-reference-ollama-seed-1"
    return atif_trajectory_to_dict(
        trajectory_id=trajectory_id,
        session_id=trajectory_id,
        agent_name="slack-ollama-installed-agent",
        agent_version="0.1.0",
        model_name=model,
        steps=steps,
        final_metrics={
            "total_steps": len(steps),
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
        },
        extra={
            "final_answer": answer,
            "expected_answer": "#platform-outages",
            "final_state_snapshot": export_state(db_path),
        },
    )


def run_ollama_agent(
    *,
    db_path: Path,
    instruction: str,
    trajectory_path: Path,
    transcript_path: Path,
    model: str,
    host: str,
    timeout_sec: float,
) -> None:
    teardown_database(db_path, Path("/app/slack_seed_snapshot.sql"))
    history: list[dict[str, Any]] = []
    answer = ""
    transcript_lines = [
        f'Task: "{instruction}"',
        "[SETUP] Seeded SQLite Slack workspace: 4 users, 3 channels, 12 messages",
        f"[AGENT] provider=ollama model={model} host={host}",
    ]
    for step_index in range(1, 31):
        action = ollama_next_action(
            model=model,
            host=host,
            instruction=instruction,
            state=export_state(db_path),
            history=history,
            timeout_sec=timeout_sec,
        )
        if "final_answer" in action:
            answer = str(action["final_answer"])
            transcript_lines.append(f"[AGENT] Answer: {answer!r}")
            break
        tool_name = str(action.get("tool_name", ""))
        input_payload = action.get("input_payload", {})
        thought = str(action.get("thought", ""))
        if not tool_name or not isinstance(input_payload, dict):
            raise RuntimeError(f"Ollama returned invalid tool action: {action!r}")
        observation = execute_tool(db_path, tool_name, input_payload)
        history.append(
            {
                "thought": thought,
                "tool_name": tool_name,
                "input_payload": input_payload,
                "observation": observation,
            }
        )
        transcript_lines.extend(
            [
                f"[AGENT step={step_index}] Thinking: {thought}",
                f"[TOOL step={step_index}] {tool_name}({json.dumps(input_payload, sort_keys=True)})",
                f"[RESULT step={step_index}] {json.dumps(observation, sort_keys=True)}",
            ]
        )

    if not answer:
        final_action = ollama_next_action(
            model=model,
            host=host,
            instruction=instruction,
            state=export_state(db_path),
            history=history,
            timeout_sec=timeout_sec,
        )
        answer = str(final_action.get("final_answer", ""))
        transcript_lines.append(f"[AGENT] Answer: {answer!r}")

    trajectory = build_trajectory_from_history(
        db_path=db_path,
        instruction=instruction,
        model=model,
        history=history,
        answer=answer,
    )
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(json.dumps(trajectory, sort_keys=True, indent=2), encoding="utf-8")
    transcript_lines.append("[EVAL] expected=#platform-outages")
    transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")


def run_reference_agent(db_path: Path, instruction: str, trajectory_path: Path, transcript_path: Path) -> None:
    teardown_database(db_path, Path("/app/slack_seed_snapshot.sql"))
    search_result = search_messages(db_path, "alice incidents")
    channel_result = get_channel_messages(db_path, "C003")
    latest = search_result["messages"][0]
    answer = first_channel_reference(latest["body"])
    trajectory = build_trajectory(db_path, instruction, search_result, channel_result, answer)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(json.dumps(trajectory, sort_keys=True, indent=2), encoding="utf-8")
    transcript_path.write_text(
        "\n".join(
            [
                f'Task: "{instruction}"',
                "[SETUP] Seeded SQLite Slack workspace: 4 users, 3 channels, 12 messages",
                '[AGENT] Thinking: I need to search for alice messages in #incidents',
                '[TOOL] search_messages(query="alice incidents")',
                f'[RESULT] Found {search_result["count"]} messages',
                '[TOOL] get_channel_messages(channel_id="C003")',
                f'[RESULT] Returned {channel_result["count"]} messages',
                f'[AGENT] Answer: "{answer}"',
                "[EVAL] expected=#platform-outages",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
