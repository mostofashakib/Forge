from __future__ import annotations

import json


def print_deterministic_report(
    name: str,
    instruction: str,
    initial_state: dict,
    tool_calls: list[dict],
    state_after_actions: dict,
    final_state: dict,
    answer_check: dict,
) -> None:
    print("\n=== DETERMINISTIC TEST REPORT ===")
    print(f"Task: {name}")
    print(f"Instruction: {instruction}")
    print("Initial State:")
    print(json.dumps(initial_state, sort_keys=True, indent=2))
    print("Actions / Tool Calls:")
    for call in tool_calls:
        print(f"  - {call['tool_name']}({json.dumps(call['input_payload'], sort_keys=True)})")
    print("State After Actions:")
    print(json.dumps(state_after_actions, sort_keys=True, indent=2))
    print("Final State:")
    print(json.dumps(final_state, sort_keys=True, indent=2))
    print("Expected vs Actual:")
    print(json.dumps(answer_check, sort_keys=True, indent=2))
