from fleet.verifiers import LayeredVerifier, VerificationSpec
from fleet.verifiers.rewardkit_checks import register_harbor_verifier
from fleet.verifiers.verifier_specs.task_manager import (
    task_manager_seed_shape,
    tm_agent_used_atif,
    tm_rl_determinism_check,
    check_task_assignee,
    check_task_has_label,
    check_task_mutation_scope,
    tm_final_answer_is_task_ids,
    tm_tool_not_called,
)

# Base (30 tasks, 13 deps, 22 assignments) + TASK031 (U004) + TASK032 (U004) = 32 tasks, 24 assignments
_EXPECTED_SEED = {"users": 6, "projects": 4, "milestones": 5, "tasks": 32, "assignments": 24, "dependencies": 13}
_EXPECTED_TASK_IDS = ["TASK006", "TASK008", "TASK009", "TASK031", "TASK032"]

# U004's tasks after seeding:
#   TASK006 (P002, M003, labels=[docs])         → has milestone → reassign to U002
#   TASK008 (P004, M005, labels=[research,ux])  → has milestone → reassign to U002
#   TASK009 (P004, M005, labels=[design,ux])    → has milestone → reassign to U002
#   TASK031 (P004, no milestone, labels=[design,frontend]) → no milestone → reassign to U003 + append needs-triage
#   TASK032 (P004, M005, labels=[design])       → has milestone → reassign to U002


def task_manager_task_1_verifier() -> LayeredVerifier:
    return LayeredVerifier(
        VerificationSpec(
            state_checks=[
                check_task_assignee("TASK006", "U002"),
                check_task_assignee("TASK008", "U002"),
                check_task_assignee("TASK009", "U002"),
                check_task_assignee("TASK031", "U003"),
                check_task_has_label("TASK031", "needs-triage"),
                check_task_has_label("TASK031", "design"),
                check_task_has_label("TASK031", "frontend"),
                check_task_assignee("TASK032", "U002"),
            ],
            invariant_checks=[
                task_manager_seed_shape(_EXPECTED_SEED),
                tm_agent_used_atif(),
                tm_rl_determinism_check(),
                check_task_mutation_scope(
                    _EXPECTED_TASK_IDS,
                    {
                        "TASK006": {"assignee_id"},
                        "TASK008": {"assignee_id"},
                        "TASK009": {"assignee_id"},
                        "TASK031": {"assignee_id", "labels"},
                        "TASK032": {"assignee_id"},
                    },
                ),
            ],
            trajectory_checks=[
                tm_final_answer_is_task_ids(_EXPECTED_TASK_IDS),
            ],
            negative_checks=[
                tm_tool_not_called("delete_task"),
            ],
        )
    )


def task_manager_task_1_reward_checks() -> list[str]:
    return [
        "state: TASK006 reassigned to U002 (has milestone M003)",
        "state: TASK008 reassigned to U002 (has milestone M005)",
        "state: TASK009 reassigned to U002 (has milestone M005)",
        "state: TASK031 reassigned to U003 (no milestone)",
        "state: TASK031 has label needs-triage (appended)",
        "state: TASK031 retains label design (not overwritten)",
        "state: TASK031 retains label frontend (not overwritten)",
        "state: TASK032 reassigned to U002 (has milestone M005)",
        "invariant: seeded task manager has expected entity counts",
        "invariant: trajectory is ATIF-v1.7 and starts with the user instruction",
        "invariant: environment reset determinism passed",
        "invariant: only requested task fields were modified",
        "trajectory: final answer lists exactly the five modified task IDs",
        "negative: agent did not call delete_task",
    ]


if not __name__.endswith("_check"):
    register_harbor_verifier(
        "check:task_manager_task_1_verifier",
        "task_manager_task_1",
        task_manager_task_1_reward_checks(),
    )
