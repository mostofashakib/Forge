#!/usr/bin/env python3
"""One tool list, three surfaces, and they cannot drift apart."""

from __future__ import annotations

import sys

from task_sim import agent_cli
from task_sim.environment import TaskHandoverEnvironment
from task_sim.service import TOOL_NAMES, _TOOL_HANDLERS
from task_sim.tool_definitions import (
    TOOL_DEFINITIONS,
    get_tool_definitions,
    tool_names,
    validate_tool_payload,
)
from workspace import Workspace

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def test_every_declared_tool_has_a_handler() -> None:
    declared = set(tool_names())
    handled = set(_TOOL_HANDLERS)
    check("no tool is advertised without an implementation", declared - handled == set(),
          str(declared - handled))
    check("no handler is unreachable from the schema", handled - declared == set(),
          str(handled - declared))
    check("TOOL_NAMES is the same list", set(TOOL_NAMES) == declared)


def test_the_cli_accepts_exactly_what_the_schemas_declare() -> None:
    """A flag the CLI takes that no tool declares is a flag that does nothing."""
    schema_fields: set[str] = set()
    for tool in TOOL_DEFINITIONS:
        schema_fields.update(tool["input_schema"]["properties"])
    cli_fields = (
        {key for key, _ in agent_cli._FLAGS.values()}
        | set(agent_cli._LIST_FLAGS.values())
        | set(agent_cli._INTEGER_FLAGS.values())
        | set(agent_cli._BOOLEAN_FLAGS.values())
    )
    check("no CLI flag maps to a field nothing declares", cli_fields - schema_fields == set(),
          str(cli_fields - schema_fields))
    check("every declared field is reachable from the CLI",
          schema_fields - cli_fields == set(), str(schema_fields - cli_fields))


def test_the_rl_environment_offers_the_same_tools() -> None:
    with Workspace() as workspace:
        environment = TaskHandoverEnvironment(workspace.db_path, workspace.snapshot_path)
        offered = {tool["name"] for tool in environment.tool_definitions()}
        check("the training surface matches the graded one", offered == set(tool_names()),
              str(offered ^ set(tool_names())))


def test_the_schemas_are_closed() -> None:
    for tool in TOOL_DEFINITIONS:
        schema = tool["input_schema"]
        check(f"{tool['name']}: additionalProperties is false",
              schema.get("additionalProperties") is False)
        check(f"{tool['name']}: every required key is declared",
              set(schema.get("required", [])) <= set(schema["properties"]))
        check(f"{tool['name']}: has a description", bool(tool["description"].strip()))


def test_validation_enforces_what_the_schema_advertises() -> None:
    check("an unknown tool is named as such",
          "Unknown tool" in (validate_tool_payload("teleport", {}) or ""))
    check("a misspelled parameter is refused",
          "Unexpected parameter" in (validate_tool_payload("get_task", {"taskid": "X"}) or ""))
    check("a missing required key is refused",
          "Missing required" in (validate_tool_payload("get_task", {}) or ""))
    check("a bad enum value is refused",
          "Invalid value" in (validate_tool_payload("list_tasks", {"status": "SHIPPED"}) or ""))
    check("enums match case-insensitively, as the tools normalize",
          validate_tool_payload("list_tasks", {"status": "pending"}) is None)
    check("a valid payload passes",
          validate_tool_payload("update_task", {"task_id": "TASK001", "assignee": "U002"}) is None)
    check("a non-object payload is refused",
          validate_tool_payload("get_task", ["TASK001"]) is not None)


def test_get_tool_definitions_hands_out_copies() -> None:
    first = get_tool_definitions()
    first[0]["name"] = "mutated"
    check("a caller cannot corrupt the surface for everyone else",
          get_tool_definitions()[0]["name"] != "mutated")


def test_every_tool_is_reachable_through_the_world() -> None:
    """Each tool runs at least once, so none is declared but broken."""
    # Ordered pairs, not a dict: create_task appears twice, and a dict literal
    # would silently keep only the second -- which is how the scratch task the
    # later steps depend on went missing.
    exercised: list[tuple[str, dict]] = [
        ("list_tasks", {}),
        ("get_task", {"task_id": "TASK001"}),
        ("list_users", {}),
        ("list_projects", {}),
        ("get_project", {"project_id": "P001"}),
        ("create_task", {"title": "Scratch", "task_id": "TASK800"}),
        ("update_task", {"task_id": "TASK800", "priority": "HIGH"}),
        ("move_task_to_project", {"task_id": "TASK800", "project_id": "P002"}),
        ("link_tasks", {"task_id": "TASK800", "depends_on_task_id": "TASK001"}),
        ("unlink_tasks", {"task_id": "TASK800", "depends_on_task_id": "TASK001"}),
        ("create_project", {"name": "Scratch project", "project_id": "P800"}),
        ("create_milestone", {"project_id": "P800", "title": "Scratch milestone"}),
        ("archive_task", {"task_id": "TASK800"}),
        ("create_task", {"title": "Scratch 2", "task_id": "TASK801"}),
        ("mark_task_duplicate", {"task_id": "TASK801", "original_task_id": "TASK001"}),
        ("delete_task", {"task_id": "TASK002"}),
        ("submit_handover_report", {"task_ids": ["TASK001"], "summary": "scratch"}),
    ]
    covered = {tool for tool, _ in exercised}
    check("the exercise covers every tool", covered == set(tool_names()),
          str(set(tool_names()) - covered))
    with Workspace() as workspace:
        for tool, payload in exercised:
            outcome = workspace.try_call(tool, **payload)
            check(f"{tool} runs", not isinstance(outcome, Exception), str(outcome))


def main() -> int:
    print(__doc__)
    for test in (
        test_every_declared_tool_has_a_handler,
        test_the_cli_accepts_exactly_what_the_schemas_declare,
        test_the_rl_environment_offers_the_same_tools,
        test_the_schemas_are_closed,
        test_validation_enforces_what_the_schema_advertises,
        test_get_tool_definitions_hands_out_copies,
        test_every_tool_is_reachable_through_the_world,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all tool-contract checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
