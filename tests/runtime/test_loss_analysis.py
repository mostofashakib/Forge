from __future__ import annotations

from forge.runtime.agent_logger import AgentRunLogger
from forge.runtime.loss_analysis import (
    FailureMode,
    LossAnalyzer,
    RunLossReport,
    cluster_failure_modes,
)
from forge.runtime.reward_hacking import AuditFinding, AuditReport
from forge.runtime.verification import CheckResult, VerificationResult


# ---------------------------------------------------------------------------
# Helpers for crafting traces and verifier results
# ---------------------------------------------------------------------------

def _logger(run_id: str, steps: list[dict], *, status: str = "completed") -> AgentRunLogger:
    """Build a trace from a compact per-step description.

    Each step dict may carry: ``response`` (an LLM final answer), ``action``
    (the tool call dict), ``result`` (the tool result), ``reward``.
    """
    logger = AgentRunLogger(run_id)
    logger.start_run()
    for i, step in enumerate(steps):
        logger.set_step(i)
        if "response" in step or "prompt" in step:
            logger.log_llm_call(
                prompt=step.get("prompt"),
                tool_call=step.get("action"),
                response=step.get("response"),
            )
        if "action" in step:
            logger.log_action(
                action=step["action"],
                result=step.get("result"),
                reward=step.get("reward"),
            )
    logger.end_run(status=status)
    return logger


def _verification(passed: bool, checks: list[CheckResult]) -> VerificationResult:
    return VerificationResult.from_checks("v", checks) if checks else VerificationResult(
        verifier_id="v", passed=passed, score=1.0 if passed else 0.0, checks=[]
    )


def _fail(name: str, evidence: str) -> CheckResult:
    return CheckResult(name=name, passed=False, score=0.0, evidence=evidence)


def _pass(name: str) -> CheckResult:
    return CheckResult(name=name, passed=True, score=1.0, evidence=None)


# ---------------------------------------------------------------------------
# One positive example per taxonomy mode
# ---------------------------------------------------------------------------

def test_instruction_following_detected_from_forbidden_action():
    # The agent saw the record in a tool result but took a forbidden action.
    logger = _logger("r", [
        {"action": {"type": "read_record", "id": "acct_1"},
         "result": {"balance": 100, "status": "frozen"}},
        {"action": {"type": "delete_record", "id": "acct_1"}, "result": {"deleted": True},
         "response": "Done."},
    ])
    verification = _verification(False, [
        _fail("trajectory:forbidden_actions", "unnecessary actions taken: ['delete_record']"),
    ])
    report = LossAnalyzer().analyze(logger, verification)
    assert FailureMode.INSTRUCTION_FOLLOWING in report.modes


def test_hallucination_detected_when_answer_names_absent_entity():
    logger = _logger("r", [
        {"action": {"type": "list_users"}, "result": {"users": ["alice", "bob"]}},
        {"action": {"type": "final"}, "response": "The account owner is Zephyr."},
    ])
    verification = _verification(False, [_fail("state:state_0", "wrong owner")])
    report = LossAnalyzer().analyze(logger, verification)
    signal = next(s for s in report.signals if s.mode is FailureMode.HALLUCINATION)
    assert "zephyr" in signal.evidence.lower()


def test_tool_sequencing_detected_from_out_of_order_check():
    logger = _logger("r", [
        {"action": {"type": "submit"}},
        {"action": {"type": "validate"}, "response": "Submitted."},
    ])
    verification = _verification(False, [
        _fail("trajectory:action_sequence",
              "required actions not in order ['validate', 'submit']"),
    ])
    report = LossAnalyzer().analyze(logger, verification)
    assert FailureMode.TOOL_SEQUENCING in report.modes


def test_early_stopping_detected_from_short_confident_wrong_run():
    logger = _logger("r", [
        {"action": {"type": "peek"}, "result": {"n": 1}, "response": "The answer is 42."},
    ])
    verification = _verification(False, [_fail("state:state_0", "wrong")])
    report = LossAnalyzer().analyze(logger, verification)
    assert report.modes == [FailureMode.EARLY_STOPPING]


def test_context_loss_detected_when_early_fact_is_dropped():
    logger = _logger("r", [
        {"action": {"type": "lookup"}, "result": {"note": "Helsinki is the capital"}},
        {"action": {"type": "lookup"}, "result": {"note": "Bergen is a coastal city"}},
        {"action": {"type": "final"}, "response": "The capital is Bergen."},
    ])
    verification = _verification(False, [_fail("state:state_0", "wrong capital")])
    report = LossAnalyzer().analyze(logger, verification, task={"expected_answer": "Helsinki"})
    modes = report.modes
    assert FailureMode.CONTEXT_LOSS in modes
    # Bergen appears in a tool result, so the answer is grounded — not a hallucination.
    assert FailureMode.HALLUCINATION not in modes


def test_reward_hacking_detected_from_audit_report_even_when_passing():
    logger = _logger("r", [
        {"action": {"type": "toggle"}, "response": "Complete."},
    ])
    verification = _verification(True, [_pass("state:state_0")])
    audit = AuditReport(flagged=True, findings=[
        AuditFinding(code="milestones_skipped", severity="high",
                     detail="verifier passed but milestones never reached: ['approved']"),
    ])
    report = LossAnalyzer().analyze(logger, verification, audit_report=audit)
    signal = next(s for s in report.signals if s.mode is FailureMode.REWARD_HACKING)
    assert "milestones" in signal.evidence
    assert signal.confidence >= 0.85  # high-severity finding


def test_surface_overfitting_detected_from_verbatim_empty_search():
    instruction = "Find the engineer who closed the most incidents last quarter"
    logger = _logger("r", [
        {"action": {"type": "search", "query": "the engineer who closed the most incidents"},
         "result": {"results": []}},
        {"action": {"type": "final"}, "response": "No one matched."},
    ])
    verification = _verification(False, [_fail("state:state_0", "no answer")])
    report = LossAnalyzer().analyze(logger, verification, task={"instruction": instruction})
    signal = next(s for s in report.signals if s.mode is FailureMode.SURFACE_OVERFITTING)
    assert "engineer who closed" in signal.evidence.lower()


# ---------------------------------------------------------------------------
# False-positive guard: a clean, correct run yields no failure modes
# ---------------------------------------------------------------------------

def test_clean_passing_run_yields_no_failure_modes():
    logger = _logger("r", [
        {"action": {"type": "list_users"}, "result": {"users": ["alice", "bob"]}},
        {"action": {"type": "read", "id": "alice"}, "result": {"role": "admin"}},
        {"action": {"type": "final"}, "response": "alice is the admin."},
    ])
    verification = _verification(True, [_pass("state:state_0"), _pass("trajectory:required_actions")])
    report = LossAnalyzer().analyze(
        logger, verification,
        task={"instruction": "Who is the admin?", "expected_answer": "alice"},
        audit_report=AuditReport(flagged=False, findings=[]),
    )
    assert report.signals == []
    assert report.passed is True


# ---------------------------------------------------------------------------
# A run exhibiting two modes is classified into both
# ---------------------------------------------------------------------------

def test_run_with_two_modes_classified_into_both():
    # Short + confident + wrong (early stopping) AND names an absent entity
    # (hallucination).
    logger = _logger("r", [
        {"action": {"type": "final"}, "response": "The manager is Zephyr."},
    ])
    verification = _verification(False, [_fail("state:state_0", "wrong")])
    report = LossAnalyzer().analyze(
        logger, verification, task={"instruction": "Who is the manager?"}
    )
    assert set(report.modes) == {FailureMode.EARLY_STOPPING, FailureMode.HALLUCINATION}


# ---------------------------------------------------------------------------
# Report shape and cross-run aggregation
# ---------------------------------------------------------------------------

def test_report_is_keyed_to_run_and_serializable():
    logger = _logger("run-42", [{"action": {"type": "final"}, "response": "The boss is Nobody."}])
    verification = _verification(False, [_fail("state:state_0", "wrong")])
    report = LossAnalyzer().analyze(logger, verification, task={"instruction": "Who?"})
    data = report.to_dict()
    assert data["run_id"] == "run-42"
    assert data["passed"] is False
    assert all({"mode", "confidence", "evidence"} <= set(s) for s in data["signals"])


def test_cluster_failure_modes_aggregates_across_runs():
    def report_with(run_id, modes):
        from forge.runtime.loss_analysis import FailureSignal
        return RunLossReport(
            run_id=run_id,
            passed=False,
            signals=[FailureSignal(mode=m, confidence=0.8, evidence="e") for m in modes],
        )

    reports = [
        report_with("a", [FailureMode.HALLUCINATION, FailureMode.EARLY_STOPPING]),
        report_with("b", [FailureMode.HALLUCINATION]),
        report_with("c", [FailureMode.HALLUCINATION]),
        report_with("d", [FailureMode.EARLY_STOPPING]),
    ]
    clusters = cluster_failure_modes(reports)
    by_mode = {c.check_name: c for c in clusters}
    assert by_mode["hallucination"].count == 3
    assert by_mode["early_stopping"].count == 2
    # Sorted most-frequent first, like FailureClusterer.
    assert clusters[0].check_name == "hallucination"
    assert set(by_mode["hallucination"].episode_ids) == {"a", "b", "c"}
