from forge.runtime.verification import CheckResult, VerificationResult


def verify_archive_newsletter(state: dict, trajectory, task: dict) -> VerificationResult:
    email_id = task["inputs"]["email_id"]
    email = state["emails"].get(email_id, {})
    passed = email.get("archived", False) is True

    return VerificationResult.from_checks(
        "archive_newsletter",
        [CheckResult(
            name="email_archived",
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence=None if passed else f"Email '{email_id}' has not been archived.",
        )],
    )
