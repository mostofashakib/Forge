from forge.runtime.verification import CheckResult, VerificationResult


def verify_escalate_billing_complaint(state: dict, trajectory, task: dict) -> VerificationResult:
    thread_id = task["inputs"]["thread_id"]
    thread = state["threads"].get(thread_id, {})
    passed = thread.get("escalated", False) is True

    return VerificationResult.from_checks(
        "escalate_billing_complaint",
        [CheckResult(
            name="thread_escalated",
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"Thread '{thread_id}' has not been escalated.",
        )],
    )
