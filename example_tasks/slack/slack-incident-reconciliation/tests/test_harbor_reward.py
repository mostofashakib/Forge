#!/usr/bin/env python3
"""The reward Harbor actually reads.

The layered verifier is only worth building if it is the thing that scores the
run. These tests cover the seam: an `EpisodeEvaluation` becoming the flat
key/value stream Harbor stores, and the entry point that assembles it for this
task. The properties that matter are that the key set never changes between a
graded run and a blank one -- an analysis that has to guess whether a missing
key means zero or means "not measured" is not an analysis -- and that partial
work is paid partially.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dynamic_scenario as scenario
import test_migration_readiness as entry
from slack_sim.service import export_state
from verifiers.harbor import blank_reward_dict, reward_dict, reward_keys


def test_a_blank_reward_declares_every_key_a_graded_one_does(root: Path) -> None:
    db = scenario.seeded(root / "keys")
    graded, _ = entry.evaluate(scenario.drive_terminal(db))
    assert set(graded) == set(entry.blank(0.0))
    assert set(blank_reward_dict(valid=0.0)) <= set(graded)
    assert all(isinstance(value, float) for value in graded.values())


def test_the_oracle_is_paid_in_full(root: Path) -> None:
    db = scenario.seeded(root / "oracle")
    rewards, details = entry.evaluate(scenario.drive_terminal(db))
    assert rewards["valid"] == 1.0
    assert rewards["reward"] == 1.0, details["evaluation"]["notes"]
    assert rewards["success"] == 1.0
    assert rewards["penalty_total"] == 0.0


def test_the_graded_answer_is_the_message_in_the_thread(root: Path) -> None:
    """The old reward routed two thirds of its weight through a JSON file whose
    path the prompt never names, which capped a correct episode at 0.42 for a
    reason that was never about the agent's work. There is no such file now:
    grading takes one argument, the world's state export.
    """
    import inspect

    signature = inspect.signature(entry.evaluate)
    assert list(signature.parameters) == ["state", "preset"], signature

    rewards, _ = entry.evaluate(scenario.drive_terminal(scenario.seeded(root / "thread")))
    assert rewards["reward"] == 1.0


def test_the_verifier_reads_nothing_the_agent_could_write(root: Path) -> None:
    """No trajectory, no report file, no working directory. The state export
    arrives over a socket the agent's user cannot open."""
    source = Path(entry.__file__).read_text(encoding="utf-8")
    for forbidden in ("load_artifact", "ARTIFACT", "/logs/agent", "TASK_WORKSPACE"):
        assert forbidden not in source, f"the verifier still references {forbidden}"


def test_partial_work_is_paid_partially(root: Path) -> None:
    db = scenario.seeded(root / "partial")
    scenario.call(db, "search_messages", {"query": "IDP-ACME-014"})
    scenario.call(db, "reply_to_thread", {"thread_parent_id": "MSG200", "body": "Looks good to me"})
    rewards, _ = entry.evaluate(export_state(db))
    assert 0.0 < rewards["reward"] < 0.5
    assert rewards["success"] == 0.0
    assert rewards["valid"] == 1.0


def test_every_layer_is_reported_separately(root: Path) -> None:
    db = scenario.seeded(root / "layers")
    rewards, _ = entry.evaluate(scenario.drive_terminal(db))
    for layer in ("final_state", "milestones", "trajectory", "negative"):
        assert f"layer_{layer}" in rewards
        assert f"layer_weight_{layer}" in rewards
    assert rewards["layer_final_state"] == 1.0


def test_a_tampered_workspace_scores_nothing_and_is_marked_invalid(root: Path) -> None:
    db = scenario.seeded(root / "tampered")
    state = scenario.drive_terminal(db)
    state["messages"] = [m for m in state["messages"] if m["message_id"] != "MSG145"]
    rewards, details = entry.evaluate(state)
    assert rewards["valid"] == 0.0
    assert rewards["reward"] == 0.0 and rewards["success"] == 0.0
    assert "integrity_error" in details


def test_missing_state_is_invalid_rather_than_a_zero_score(root: Path) -> None:
    rewards, details = entry.evaluate(None)
    assert rewards["valid"] == 0.0
    assert set(rewards) == set(entry.blank(0.0))
    assert "integrity_error" in details


def test_the_preset_is_recorded_alongside_the_score_it_produced(root: Path) -> None:
    db = scenario.seeded(root / "preset")
    state = scenario.drive_terminal(db)
    layered, _ = entry.evaluate(state, preset="full_layered_deterministic")
    binary, details = entry.evaluate(state, preset="binary_final_state")
    assert details["evaluation"]["preset"] == "binary_final_state"
    assert binary["reward"] in (0.0, 1.0)
    assert layered["reward"] == 1.0


def test_the_details_carry_the_check_by_check_account(root: Path) -> None:
    db = scenario.seeded(root / "details")
    _, details = entry.evaluate(scenario.drive_terminal(db))
    names = {
        check["name"]
        for result in details["evaluation"]["results"]
        for check in result["checks"]
    }
    assert "audienceMatches_critiqued" in names
    assert "cites_sso_evidence" in names
    # details.json is written verbatim; anything unserialisable there is a
    # verifier that reports nothing at all.
    json.dumps(details, default=str)


def test_the_key_stream_is_the_layered_reward_and_nothing_else(root: Path) -> None:
    """The dimension/failure/milestone keys are gone.

    They came from a scorer that read the agent's own report artifact and
    carried a second definition of stale evidence that had already drifted from
    the live one. Two graders in one key stream is two answers to "how did this
    run do", and only one of them was the reward.
    """
    db = scenario.seeded(root / "keys-only")
    rewards, details = entry.evaluate(scenario.drive_terminal(db))
    stray = [key for key in rewards
             if key.startswith(("dimension_", "failure_", "milestone_", "legacy_"))]
    assert not stray, stray
    assert "legacy" not in details
    assert not hasattr(entry, "LEGACY_KEYS"), "the compatibility hook is still there"
    assert set(rewards) == set(reward_keys()), "the key stream is not the layered one"


def test_reward_dict_is_a_flat_map_of_floats(root: Path) -> None:
    from verifiers.results import LayerScore, RewardBreakdown, EpisodeEvaluation

    evaluation = EpisodeEvaluation(
        preset="full_layered_deterministic",
        passed=False,
        reward=0.25,
        breakdown=RewardBreakdown(layers=(LayerScore("final_state", 1.0, 0.25),), base=0.25),
    )
    flat = reward_dict(evaluation)
    assert flat["reward"] == 0.25 and flat["success"] == 0.0 and flat["valid"] == 1.0
    assert flat["layer_final_state"] == 0.25
    assert all(isinstance(value, float) for value in flat.values())


def test_grading_never_reads_the_agent_authored_artifact(root: Path) -> None:
    """The layered stack grades world state the agent cannot write. A legacy
    scorer that read the report artifact used to run alongside it and could
    invalidate the episode, which put an agent-authored file back on the path
    that decides whether a run counts."""
    import ast

    import test_migration_readiness as entry

    source = Path(entry.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if (node.module or "").endswith("migration_reward")
    }
    # Two names, and the integrity gate is the reason for both: it raises, and
    # the caller has to name what it catches. What must not come back is any
    # part of the scorer itself.
    #
    # `ARTIFACT_PATH` used to be permitted here for an optional agent-written
    # report. That report is gone, nothing defines the name, and
    # `test_the_verifier_reads_nothing_the_agent_could_write` above forbids the
    # very substring -- so allowing it made these two checks disagree.
    allowed = {"check_workspace_integrity", "IntegrityError"}
    assert imported <= allowed, (
        f"the verifier still pulls {sorted(imported - allowed)} from the legacy scorer"
    )
    assert "grade" not in imported


def test_a_tampered_workspace_is_still_caught_without_the_legacy_scorer(root: Path) -> None:
    from slack_sim.migration_reward import IntegrityError, check_workspace_integrity

    db = scenario.seeded(root / "tampered")
    scenario.drive_terminal(db)
    state = export_state(db)
    state["messages"] = [m for m in state["messages"] if m["message_id"] != "MSG135"]
    try:
        check_workspace_integrity(state)
    except IntegrityError:
        return
    raise AssertionError("destroyed seeded history passed the integrity check")


def test_a_clean_workspace_passes_the_integrity_check(root: Path) -> None:
    from slack_sim.migration_reward import check_workspace_integrity

    db = scenario.seeded(root / "clean")
    scenario.drive_terminal(db)
    check_workspace_integrity(export_state(db))


def test_the_command_line_grader_reports_the_same_number(root: Path) -> None:
    """`python3 -m verifiers.run` is a real entry point that nothing imports,
    so nothing caught it when the contract's helpers were renamed under it."""
    import subprocess

    db = scenario.seeded(root / "cli")
    state_file = root / "state.json"
    state_file.write_text(json.dumps(scenario.drive_terminal(db)), encoding="utf-8")

    package_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "verifiers.run", "--state", str(state_file)],
        capture_output=True, text=True, cwd=package_root,
        env={"PYTHONPATH": f"{package_root / 'environment'}:{package_root}", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert "reward=1.0000" in result.stdout, result.stdout


def main() -> None:
    tests = sorted(
        (value for name, value in globals().items()
         if name.startswith("test_") and callable(value)),
        key=lambda fn: fn.__name__,
    )
    with tempfile.TemporaryDirectory() as directory:
        for index, test in enumerate(tests):
            path = Path(directory) / str(index)
            path.mkdir()
            test(path)
            print(f"  {test.__name__}: ok")
    print(f"harbor reward: ok ({len(tests)} tests)")


if __name__ == "__main__":
    main()
