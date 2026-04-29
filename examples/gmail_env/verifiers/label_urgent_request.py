from forge.runtime.verification import CheckResult, VerificationResult


def verify_label_urgent_request(state: dict, trajectory, task: dict) -> VerificationResult:
    email_id = task["inputs"]["email_id"]
    email = state["emails"].get(email_id, {})
    passed = "urgent" in email.get("labels", [])

    return VerificationResult.from_checks(
        "label_urgent_request",
        [CheckResult(
            name="urgent_label_applied",
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"Email '{email_id}' does not have 'urgent' label.",
        )],
    )
