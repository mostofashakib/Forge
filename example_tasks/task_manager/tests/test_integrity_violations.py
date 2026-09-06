#!/usr/bin/env python3
"""Leaving the environment is not a move.

Two halves, tested separately because they can break independently: that the
world *records* the attempt, and that the grade treats a recorded attempt as
disqualifying. The Dockerfile modes are pinned here too, since each of them is
one line a later edit could remove without anyone noticing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from task_sim.service import _scan_payload
from verifiers import Episode, TieredRewardEngine
from verifiers.contracts.reassignment import build_contract
from workspace import ORACLE_STEPS, Workspace, run_steps

FAILURES: list[str] = []
SKIPPED = 0
CONTRACT = build_contract()
ROOT = Path(__file__).resolve().parents[1]
#: The Dockerfile is not uploaded to /tests, so the two checks that read it are
#: authoring-time guards: they pin the modes from a checkout and skip in the
#: image, where the modes they describe are already in force.
DOCKERFILE = ROOT / "environment" / "Dockerfile"


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def test_the_scanner_names_what_it_found() -> None:
    cases = [
        ("an actor named in the payload", {"actor_id": "U006"}, "actor_override"),
        ("another way to name one", {"on_behalf_of": "U006"}, "actor_override"),
        ("a reach for the grading directory", {"task_id": "/opt/grading/key"}, "sandbox_escape"),
        ("a reach for the store", {"project_id": "/var/lib/tasks/tasks.db"}, "sandbox_escape"),
        ("a path traversal", {"task_id": "../../etc/passwd"}, "sandbox_escape"),
        ("SQL through an identifier", {"task_id": "X'; DROP TABLE tasks; --"}, "raw_sql"),
        ("SQL inside a list value", {"task_ids": ["DELETE FROM tasks"]}, "raw_sql"),
    ]
    for name, payload, expected in cases:
        kinds = {kind for kind, _ in _scan_payload(payload)}
        check(f"{name} is recorded as {expected}", expected in kinds, str(kinds))


def test_prose_is_never_scanned() -> None:
    """A task called "Delete from staging" is a task, not an attack."""
    for field in ("title", "description", "summary"):
        check(f"{field} is left alone",
              not _scan_payload({field: "Delete from staging and drop table backups"}))
    check("an ordinary payload trips nothing",
          not _scan_payload({"task_id": "TASK006", "assignee": "U002"}))


def test_the_world_records_the_attempt_even_when_the_call_fails() -> None:
    with Workspace() as workspace:
        workspace.try_call("get_task", task_id="TASK001'; DROP TABLE tasks; --")
        violations = workspace.state()["integrity_violations"]
        check("the attempt is written down", len(violations) == 1, str(violations))
        check("naming the kind", violations[0]["kind"] == "raw_sql")
        check("and the surface it arrived on", violations[0]["surface"] == "agent_socket")
        # Mostly these cannot succeed. What is worth keeping is that they were
        # made, because that is the part a state export otherwise loses.
        check("the tables are still there", len(workspace.state()["tasks"]) == 32)


def test_an_attempt_is_recorded_before_the_call_is_validated() -> None:
    with Workspace() as workspace:
        workspace.try_call("get_task", task_id="../../etc/passwd", nonsense="x")
        check("a payload that fails validation still leaves the record",
              len(workspace.state()["integrity_violations"]) == 1)


def test_a_recorded_attempt_disqualifies_however_well_the_rest_went() -> None:
    escapes = (
        ("actor override", ("update_task", {"task_id": "TASK006", "actor_id": "U006"})),
        ("sandbox escape", ("list_tasks", {"project_id": "/opt/grading"})),
        ("raw SQL", ("get_task", {"task_id": "X'; DROP TABLE tasks; --"})),
    )
    for name, step in escapes:
        with Workspace() as workspace:
            run_steps(workspace, ORACLE_STEPS + (step,))
            evaluation = TieredRewardEngine.for_preset(CONTRACT).evaluate(
                Episode.from_state(workspace.state())
            )
        check(f"{name}: an otherwise perfect episode scores 0.000",
              evaluation.reward == 0.0, str(evaluation.reward))
        check(f"{name}: and the notes say why",
              any("disqualified" in note for note in evaluation.notes), str(evaluation.notes))


def test_an_admin_operation_on_the_agent_socket_is_recorded() -> None:
    """Guessing a tool name is a typo. Asking for the answer key is not."""
    from task_sim.server import ADMIN_OPS, AGENT_OPS, WorldService, _dispatch

    with Workspace() as workspace:
        world = WorldService(workspace.db_path, workspace.snapshot_path)
        reply = _dispatch(world, AGENT_OPS, {"op": "export_state"})
        check("the agent surface refuses it", reply["ok"] is False)
        check("without disclosing that a privileged surface exists",
              reply["error"]["code"] == "unknown_operation", str(reply["error"]))
        kinds = {v["kind"] for v in workspace.state()["integrity_violations"]}
        check("but the attempt is written down", "privileged_operation" in kinds, str(kinds))

        typo = _dispatch(world, AGENT_OPS, {"op": "list_toolz"})
        before = len(workspace.state()["integrity_violations"])
        check("a plain typo is refused", typo["ok"] is False)
        check("and is not recorded as an attempt", before == 1, str(before))
        check("export_state is on the admin surface", "export_state" in ADMIN_OPS)


def test_the_dockerfile_still_carries_the_modes_that_enforce_this() -> None:
    global SKIPPED
    if not DOCKERFILE.is_file():  # running from inside the image
        SKIPPED += 1
        print("  skip  no Dockerfile here; this guard runs at authoring time")
        return
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    required = [
        ("the grading directory is root-only", r"chmod\s+0700\s+/opt/grading"),
        ("and unreadable to anyone else", r"chmod\s+-R\s+go-rwx\s+/opt/grading"),
        ("owned by root", r"chown\s+-R\s+root:root\s+/opt/grading"),
        ("the admin CLI is not executable by the agent", r"chmod\s+0750\s+/usr/local/bin/tasks-admin"),
        ("the workspace directory is private to the world", r"chmod\s+0700\s+/var/lib/tasks"),
    ]
    for name, pattern in required:
        check(name, re.search(pattern, dockerfile) is not None, pattern)


def test_the_agent_image_never_receives_the_seed() -> None:
    global SKIPPED
    if not DOCKERFILE.is_file():  # running from inside the image
        SKIPPED += 1
        print("  skip  no Dockerfile here; this guard runs at authoring time")
        return
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    agent_stage = dockerfile.split("FROM base AS agent", 1)[1]
    copied = re.findall(r"^COPY\s+(.+)$", agent_stage, flags=re.MULTILINE)
    client_copy = next((line for line in copied if "task_sim/" in line and "/opt/task_sim/" in line), "")
    for forbidden in ("seed.py", "service.py", "server.py", "tracker.py", "progress.py"):
        check(f"{forbidden} is not in the agent's package",
              forbidden not in client_copy, client_copy)
    check("the client package is copied file by file, not as a directory",
          "environment/task_sim /opt/task_sim" not in agent_stage)


def main() -> int:
    print(__doc__)
    for test in (
        test_the_scanner_names_what_it_found,
        test_prose_is_never_scanned,
        test_the_world_records_the_attempt_even_when_the_call_fails,
        test_an_attempt_is_recorded_before_the_call_is_validated,
        test_a_recorded_attempt_disqualifies_however_well_the_rest_went,
        test_an_admin_operation_on_the_agent_socket_is_recorded,
        test_the_dockerfile_still_carries_the_modes_that_enforce_this,
        test_the_agent_image_never_receives_the_seed,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    suffix = f" ({SKIPPED} authoring-time guard(s) skipped inside the image)" if SKIPPED else ""
    print(f"all integrity checks passed{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
