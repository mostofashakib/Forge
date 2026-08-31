from __future__ import annotations

import json
from pathlib import Path

from tests.simulation_driver import legacy_artifacts, legacy_events, reset_determinism_check

SMOKE_TRANSCRIPT_PATH = Path("/tmp/fleet_smoke/test_smoke_tasks.txt")
_TRANSCRIPT_INITIALIZED = False


def initialize_smoke_transcript() -> None:
    global _TRANSCRIPT_INITIALIZED
    if _TRANSCRIPT_INITIALIZED:
        return
    SMOKE_TRANSCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_TRANSCRIPT_PATH.write_text("=== SMOKE TEST TRANSCRIPT ===\n\n", encoding="utf-8")
    _TRANSCRIPT_INITIALIZED = True


def print_task_report(task, trajectory, verifier_results) -> None:
    events = legacy_events(trajectory)
    artifacts = legacy_artifacts(trajectory)
    answer_event = next(event for event in events if event["event_type"] == "agent_answer")
    eval_event = next(event for event in events if event["event_type"] == "eval")
    lines = [
        "",
        "=== SMOKE TASK REPORT ===",
        f"Environment: {task.environment_name}",
        f"Instruction: {task.instruction}",
        f"Expected Answer: {task.expected_answer}",
        f"Actual Answer: {answer_event['payload']['answer']}",
        f"Answer Match: {eval_event['payload']['passed']}",
        "Tool Calls:",
    ]
    for call in artifacts["tool_calls"]:
        lines.append(f"  - {call['tool_name']}({json.dumps(call['input_payload'], sort_keys=True)})")
    lines.append("Verifier Results:")
    for result in verifier_results:
        lines.append(
            f"  - layer={result.layer} passed={result.passed} "
            f"score={result.score} message={result.message} details={result.details}"
        )
    lines.extend(
        [
            "Final State:",
            json.dumps(artifacts["final_state_snapshot"], sort_keys=True, indent=2),
            f"Reset Determinism: {reset_determinism_check(trajectory)}",
            f"Transcript: {SMOKE_TRANSCRIPT_PATH}",
        ]
    )
    text = "\n".join(lines)
    print(text)
    with SMOKE_TRANSCRIPT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")
