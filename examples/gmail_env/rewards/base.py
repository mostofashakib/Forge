from forge.contracts import RewardBreakdown, Rubric
from forge.runtime.reward import RewardComponent


def compute_gmail_reward(
    state: dict,
    trajectory,
    verifier_results: list,
    task: dict | None = None,
) -> RewardBreakdown:
    pass_rate = (
        sum(vr.score for vr in verifier_results) / len(verifier_results)
        if verifier_results
        else 0.0
    )
    step_count = trajectory.step_count
    has_violation = trajectory.has_policy_violation

    task_success = pass_rate * 1.0
    step_penalty = 0.01 * step_count
    violation_penalty = 1.0 if has_violation else 0.0

    total = task_success - step_penalty - violation_penalty
    total = max(-1.0, min(1.0, total))

    return RewardBreakdown(
        total_reward=total,
        components=[
            RewardComponent(name="task_success", value=task_success),
            RewardComponent(name="step_efficiency", value=-step_penalty),
            RewardComponent(name="policy_compliance", value=-violation_penalty),
        ],
    )


class GmailRubric(Rubric):
    """Contract form of the generated rubric.

    Gmail has a single rubric shared across all four tasks, so unlike the
    per-task template output there is one class here rather than one per
    task — the plain function above stays the implementation.
    """

    def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:
        return compute_gmail_reward(state, trajectory, verifier_results, task)
