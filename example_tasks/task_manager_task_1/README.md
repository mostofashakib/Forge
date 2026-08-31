# Task Manager Task 1: Conditional Reassignment

This Harbor task evaluates conditional planning over a deterministic SQLite task tracker. The agent must find every task currently assigned to Jordan Kim, branch on milestone presence, preserve existing labels, and avoid unrelated mutations.

## What the task measures

- Enumerating the complete target set before mutating state
- Reassigning milestone tasks to Morgan Patel
- Reassigning milestone-free tasks to Riley Stone
- Appending `needs-triage` without replacing existing labels
- Preserving all unrelated tasks and task fields
- Returning exactly the complete set of modified task IDs

## Run it

From the `example_tasks` root:

```bash
./run.sh validate
./run.sh setup
./run.sh task-manager-task-1
```

The default model is `gemma4:26b`. Override it with `FLEET_MODEL=<model>`.

Harbor builds the environment and verifier images from this directory's Docker Compose files. Results are written under `jobs/<timestamp>/` and include the ATIF JSON trajectory, text transcript, verifier report, and per-criterion rewards.

## Files

- `instruction.md`: conditional reassignment and final-answer contract
- `environment/task_manager_env.py`: thin CLI wrapper around the shared service
- `environment/Dockerfile`: deterministic base seed plus task-specific fixtures
- `solution/solve.sh`: known-good inspect-then-update workflow
- `tests/check.py`: state, mutation-scope, invariant, final-answer, and negative checks
- `task.toml`: Harbor resource limits and artifact declarations

The shared models, schemas, service implementation, and verifier helpers live under `fleet/`; they are not copied into the task.
