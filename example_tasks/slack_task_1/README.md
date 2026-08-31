# Slack Task 1: Identity-Sensitive Thread Operations

This Harbor task evaluates a multi-step Slack workflow in a deterministic SQLite workspace. The agent acts as Ben Ortiz and must distinguish DMs, group chats, channels, thread parents, and a look-alike user named Ben Ortíz.

## What the task measures

- Selecting existing chats instead of creating replacements
- Routing messages through DM and group-specific tools
- Preserving the required DM-before-group action order
- Editing only the acting user's thread reply
- Deriving reply text from the latest thread commenter
- Returning an exact semicolon-delimited final answer
- Avoiding unrelated channel, profile, reaction, and deletion mutations

## Run it

From the `example_tasks` root:

```bash
./run.sh validate
./run.sh setup
./run.sh slack-task-1
```

The default model is `gemma4:26b`. Override it with `FLEET_MODEL=<model>`.

Harbor builds the environment and verifier images from this directory's Docker Compose files. Results are written under `jobs/<timestamp>/` and include the ATIF JSON trajectory, text transcript, verifier report, and per-criterion rewards.

## Files

- `instruction.md`: agent-facing task contract
- `environment/slack_env.py`: thin CLI wrapper around `fleet.environments.slack`
- `solution/solve.sh`: known-good deterministic workflow
- `tests/check.py`: state, invariant, trajectory, and negative verifier checks
- `task.toml`: Harbor resource limits and artifact declarations

The shared seed, tool schemas, service implementation, and verifier helpers live under `fleet/`; they are not copied into the task.
