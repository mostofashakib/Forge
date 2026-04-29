from forge.runtime.verification import CheckResult, VerificationResult


def verify_reply_to_customer(state: dict, trajectory, task: dict) -> VerificationResult:
    thread_id = task["inputs"]["thread_id"]

    replies = [
        e for e in trajectory.events
        if e["type"] == "email_replied" and e["entity_id"] == thread_id
    ]

    passed = len(replies) > 0
    return VerificationResult.from_checks(
        "reply_to_customer",
        [CheckResult(
            name="reply_sent",
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"No reply sent to thread '{thread_id}'.",
        )],
    )
