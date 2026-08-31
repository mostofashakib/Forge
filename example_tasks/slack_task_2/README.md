# Slack Task 2: Channel and Group Name Collision

This Harbor task evaluates whether an agent can preserve entity identity through a multi-step Slack workflow. The seeded public channel and newly created group intentionally share the name `security-alerts`.

## What the task measures

- Creating exactly one group with the requested participants
- Posting to the existing public channel rather than the same-named group
- Replying to the correct thread and reacting to the created reply
- Sending the welcome message to the created group
- Renaming that same group after sending the message
- Returning only the exact created group ID
- Preserving the required action order and avoiding unrelated mutations

## Run it

From the `example_tasks` root:

```bash
./run.sh validate
./run.sh setup
./run.sh slack-task-2
```

The default model is `gemma4:26b`. Override it with `FLEET_MODEL=<model>`.

Harbor builds the environment and verifier images from this directory's Docker Compose files. Results are written under `jobs/<timestamp>/` and include the ATIF JSON trajectory, text transcript, verifier report, and per-criterion rewards.

## Files

- `instruction.md`: explicit channel/group identity and ordering contract
- `environment/slack_env.py`: thin CLI wrapper around `fleet.environments.slack`
- `solution/solve.sh`: known-good solution that carries dynamic group/reply IDs forward
- `tests/check.py`: state, invariant, action-order, final-answer, and negative checks
- `task.toml`: Harbor resource limits and artifact declarations

The shared seed, tool schemas, service implementation, and verifier helpers live under `fleet/`; they are not copied into the task.
