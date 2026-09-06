#!/usr/bin/env python3
"""The grading stack: the five verifier types, the four layers, the veto."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from verifiers import (
    ActionPenalty, Contract, Episode, EventVerifier, ExactStateVerifier, Forbidden,
    LayeredVerifier, NegativeVerifier, Ordering, Policy, PolicyVerifier,
    RewardHackingAuditor, StateExpectation, TemporalVerifier, TieredRewardEngine,
    ToolExpectation, TrajectoryVerifier,
)
from verifiers.contracts.reassignment import MILESTONE_EVENTS, build_contract
from verifiers.results import LAYERS
from workspace import ORACLE_STEPS, Workspace, oracle_state, run_steps

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def episode_for(steps) -> Episode:
    with Workspace() as workspace:
        run_steps(workspace, steps)
        return Episode.from_state(workspace.state())


def test_the_contract_covers_every_layer() -> None:
    contract = build_contract()
    for layer in LAYERS:
        check(f"{layer} has verifiers", bool(getattr(contract, layer)))
    check("the milestone list is declared once",
          contract.milestone_events == MILESTONE_EVENTS)


def test_the_oracle_satisfies_the_contract_completely() -> None:
    """A contract that fails a known-correct episode is wrong about the task."""
    evaluation = TieredRewardEngine.for_preset(build_contract()).evaluate(
        Episode.from_state(oracle_state())
    )
    check("every layer scores 1.0",
          all(layer.score == 1.0 for layer in evaluation.breakdown.layers),
          evaluation.summary())
    check("no check failed",
          not [c.name for r in evaluation.results for c in r.checks if not c.passed],
          str([c.name for r in evaluation.results for c in r.checks if not c.passed]))


def test_exact_state_reports_what_it_saw() -> None:
    episode = episode_for(ORACLE_STEPS)
    verifier = ExactStateVerifier(expectations=(
        StateExpectation("right", lambda e: e.assignee("TASK006"), "U002"),
        StateExpectation("wrong", lambda e: e.assignee("TASK006"), "U005"),
        StateExpectation("broken", lambda e: e.nope(), "x"),
    ))
    result = verifier.verify(episode)
    outcomes = {c.name: c for c in result.checks}
    check("a satisfied expectation passes", outcomes["right"].passed)
    check("an unsatisfied one fails", not outcomes["wrong"].passed)
    check("and reports the actual value", outcomes["wrong"].detail["actual"] == "U002")
    check("a broken selector is a failed check, not a crash",
          not outcomes["broken"].passed and "error" in outcomes["broken"].detail)


def test_policy_expressions_cannot_reach_the_filesystem() -> None:
    episode = episode_for(ORACLE_STEPS)
    verifier = PolicyVerifier(policies=(
        Policy("holds", "len(tasks) > 0"),
        Policy("escape", "__import__('os').listdir('/')"),
        Policy("typo", "tasksss"),
    ))
    outcomes = {c.name: c for c in verifier.verify(episode).checks}
    check("a true expression passes", outcomes["holds"].passed)
    check("builtins are stripped, so an escape fails as a check",
          not outcomes["escape"].passed and "error" in outcomes["escape"].detail)
    check("a typo fails as a check rather than reaching anything",
          not outcomes["typo"].passed)


def test_events_pay_for_partial_progress() -> None:
    partial = episode_for(ORACLE_STEPS[:5])
    verifier = EventVerifier(required=MILESTONE_EVENTS)
    score = verifier.verify(partial).score
    check("some milestones reached scores between the ends", 0.0 < score < 1.0, str(score))
    check("all of them scores 1.0",
          verifier.verify(episode_for(ORACLE_STEPS)).score == 1.0)
    check("none of them scores 0.0", verifier.verify(episode_for(())).score == 0.0)


def test_a_vacuous_ordering_is_not_a_pass() -> None:
    """Constraints nothing came close to violating measure nothing."""
    verifier = TemporalVerifier(orderings=(Ordering("roster_listed", "task031_reassigned"),))
    result = verifier.verify(episode_for(()))
    check("an episode that triggered nothing makes the layer unavailable",
          result.available is False)
    check("rather than scoring it a pass", result.score == 0.0)


def test_ordering_catches_an_answer_given_too_early() -> None:
    early = (
        ("list_tasks", {"assignee": "U004"}),
        ("submit_handover_report", {"task_ids": ["TASK006"], "summary": "done"}),
        ("update_task", {"task_id": "TASK006", "assignee": "U002"}),
    )
    verifier = build_contract().milestones[1]
    result = verifier.verify(episode_for(early))
    check("reporting before the work is caught",
          any(not c.passed and c.name.startswith("answered_after") for c in result.checks),
          str([c.name for c in result.checks if not c.passed]))


def test_the_negative_layer_separates_a_charge_from_a_veto() -> None:
    episode = episode_for(ORACLE_STEPS)
    verifier = NegativeVerifier(forbidden=(
        Forbidden("priced", lambda e: ["a", "b"], charge=lambda _: 0.05),
        Forbidden("fatal", lambda e: ["x"]),
        Forbidden("clean", lambda e: []),
        Forbidden("broken", lambda e: (_ for _ in ()).throw(RuntimeError("boom"))),
    ))
    result = verifier.verify(episode)
    outcomes = {c.name: c for c in result.checks}
    check("a broken detector is a failed check, never a clean episode",
          not outcomes["broken"].passed)
    # A detector that raised answered nothing, so the layer reports that it
    # could not be run: a bug in the grader must not veto an agent, and a rule
    # whose detector crashed must not read as a rule that found nothing.
    check("and it makes the whole layer unavailable", result.available is False)
    check("so nothing is disqualified on a grader bug", result.disqualifying_checks == ())
    check("and no charge is invented from it", result.penalties == ())

    working = NegativeVerifier(forbidden=(
        Forbidden("priced", lambda e: ["a", "b"], charge=lambda _: 0.05),
        Forbidden("fatal", lambda e: ["x"]),
        Forbidden("clean", lambda e: []),
    )).verify(episode)
    check("a priced rule reports penalties",
          [p for p in working.penalties if p.source == "priced"])
    check("charging once per occurrence", len(working.penalties) == 2, str(len(working.penalties)))
    check("a fatal rule reports a disqualification",
          working.disqualifying_checks == ("fatal",), str(working.disqualifying_checks))
    check("a clean rule reports neither",
          {c.name for c in working.checks if c.passed} >= {"clean"})


def test_a_veto_takes_exactly_the_credit_that_was_earned() -> None:
    dishonest = ORACLE_STEPS + (("delete_task", {"task_id": "TASK022"}),)
    evaluation = TieredRewardEngine.for_preset(build_contract()).evaluate(episode_for(dishonest))
    check("the reward is zero", evaluation.reward == 0.0, str(evaluation.reward))
    check("base minus penalties equals the total",
          abs(evaluation.breakdown.base - evaluation.breakdown.penalty_total
              - evaluation.breakdown.total) < 1e-9)
    forbidden = [p for p in evaluation.breakdown.penalties if p.source == "forbidden"]
    check("one charge names every violation, not one entry each",
          len(forbidden) == 1, str(len(forbidden)))


def test_trajectory_is_unavailable_without_an_action_log() -> None:
    """A grading gap is not an agent that did nothing."""
    state = oracle_state()
    del state["action_log"]
    result = TrajectoryVerifier(
        necessary=(ToolExpectation("anything", lambda e: True),)
    ).verify(Episode.from_state(state))
    check("the layer reports itself unavailable", result.available is False)
    check("and says why", "action log" in str(result.checks[0].detail))


def test_a_lost_layer_redistributes_its_weight() -> None:
    verifier = LayeredVerifier(
        name="t",
        verifiers=(ExactStateVerifier(expectations=(
            StateExpectation("ok", lambda e: 1, 1),
        )),),
    )
    layers = {l.layer: l for l in verifier.layer_scores(verifier.run(episode_for(())))}
    check("a layer with no verifiers is unavailable", layers["milestones"].available is False)
    check("and the surviving layer absorbs the weight",
          layers["final_state"].weight > 0.40, str(layers["final_state"].weight))


def test_the_auditor_inherits_its_milestones() -> None:
    verifier = LayeredVerifier.from_contract(build_contract())
    auditor = RewardHackingAuditor.for_verifier(verifier)
    check("the auditor cannot drift from the grader",
          auditor.milestones == verifier.milestone_events)
    report = auditor.audit(episode_for(ORACLE_STEPS), passed=True)
    check("a correct episode raises no findings", not report.findings,
          str([f.code for f in report.findings]))


def test_the_auditor_flags_a_pass_that_skipped_milestones() -> None:
    auditor = RewardHackingAuditor(milestones=("never_happens",))
    report = auditor.audit(episode_for(ORACLE_STEPS), passed=True)
    check("a milestone nobody reached is flagged",
          any(f.code == "skipped_milestone" for f in report.findings))
    check("and the audit does not call it a pass", report.audited_pass is False)


def test_no_grading_module_reads_the_agents_own_files() -> None:
    """The trajectory the agent writes is evidence written by the graded party."""
    root = Path(__file__).resolve().parents[1] / "verifiers"
    needles = ("/logs/agent", "trajectory.json", "trajectory.txt")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Docstrings are prose about the rule and are allowed to name the path
        # they exist to forbid; what must not appear is a literal the code could
        # actually open.
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and any(needle in node.value for needle in needles)):
                offenders.append(f"{path.name}:{node.value[:40]}")
    check("no verifier module can open an agent-writable path", not offenders, str(offenders))


def test_the_contract_is_the_only_module_that_knows_the_task() -> None:
    root = Path(__file__).resolve().parents[1] / "verifiers"
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = getattr(node, "module", None)
            if isinstance(node, ast.ImportFrom) and module and module.startswith("task_sim"):
                offenders.append(f"{path.name} imports {module}")
    check("the framework knows nothing about this task", not offenders, str(offenders))


def main() -> int:
    print(__doc__)
    for test in (
        test_the_contract_covers_every_layer,
        test_the_oracle_satisfies_the_contract_completely,
        test_exact_state_reports_what_it_saw,
        test_policy_expressions_cannot_reach_the_filesystem,
        test_events_pay_for_partial_progress,
        test_a_vacuous_ordering_is_not_a_pass,
        test_ordering_catches_an_answer_given_too_early,
        test_the_negative_layer_separates_a_charge_from_a_veto,
        test_a_veto_takes_exactly_the_credit_that_was_earned,
        test_trajectory_is_unavailable_without_an_action_log,
        test_a_lost_layer_redistributes_its_weight,
        test_the_auditor_inherits_its_milestones,
        test_the_auditor_flags_a_pass_that_skipped_milestones,
        test_no_grading_module_reads_the_agents_own_files,
        test_the_contract_is_the_only_module_that_knows_the_task,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all verifier checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
