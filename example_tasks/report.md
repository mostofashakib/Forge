# RL Environment Evaluation Report

## Project overview

This project provides two simulated knowledge-work environments for evaluating how reliably AI agents use workspace tools:

- **Slack:** Agents can create chats, post messages, reply in threads, add reactions, and update user profiles.
- **Task Manager:** Agents can list, create, update, delete, and archive project tasks.

Both services run in isolated, deterministic environments. This keeps evaluations safe and makes failed runs reproducible.

## Architecture and verification

Shared database logic and verification code live in the `fleet` package. Individual task directories remain small and contain only the configuration and logic specific to that task.

Each evaluation uses four layers of verification:

1. **State checks** inspect the final database state.
2. **Invariant checks** confirm that seeded data and the trajectory format remain intact.
3. **Trajectory checks** verify required tool calls and the final answer.
4. **Negative checks** reject forbidden actions.

Together, these checks prevent an agent from earning credit by reporting success without making the required state changes.

## Current limitations

The evaluated smaller models struggle most with long, multi-step tasks. As the interaction history grows, they may omit an item, stop early, or violate a strict output format even when the necessary information remains in context. All three runs below show some form of this behavior.

Scoring is currently all-or-nothing. A run that passes 11 of 12 checks receives the same final reward as a run that passes none. Because each criterion already records a separate result, weighted partial credit could provide a more useful training signal.

The environments are API- and tool-driven. Extending them to graphical interfaces would require additional controls to preserve determinism, including mocked backend calls, disabled animations, fixed clocks, and stable layouts. Without those controls, screenshots, element positions, and action traces could vary between runs.

Potential improvements include:

- recovery loops for malformed tool calls and output;
- prompts that better support long action sequences; and
- weighted scoring based on individual verifier criteria.

## Evaluation summary

| Environment | Task | Result | Primary failure |
| --- | --- | --- | --- |
| Slack | Task 1 | 31/34 checks passed | Stopped before the final reply and selected the wrong last commenter |
| Slack | Task 2 | Failed | Confused a channel with a same-named group chat |
| Task Manager | Task 1 | 11/12 checks passed | Skipped one task from a complete list |

## Slack environment

### Task 1: Thread state and look-alike identities

The agent must send the same incident-review message to a direct message and a group chat, edit its own reply in `#platform-outages`, and respond to threads in `#incidents` and `#general`. Each thread response must use the display name of its last commenter. The final answer must contain both names separated by a semicolon.

**Result:** 31 of 34 checks passed.

The agent completed the direct message, group message, reply edit, and `#incidents` response correctly. It edited `MSG023` while preserving the neighboring user reply, then posted `Welcome Alice Nguyen` in the correct incident thread.

The failure occurred in `#general`. The context contained two replies under `MSG010`: `MSG025` from Cara Singh, followed by `MSG026` from Ben Ortíz. Because “Ben Ortíz” closely resembles the agent identity “Ben Ortiz,” the model appears to have treated the last reply as its own and selected the earlier commenter. It then stopped without posting the required reply.

- **Expected answer:** `Alice Nguyen;Ben Ortíz`
- **Agent answer:** `Alice Nguyen;Cara Singh`
- **Failed checks:** `MSG010` reply state, reply trajectory, and final answer

### Task 2: Channel and group name collision

The agent must create a group chat named `security-alerts`, post to the existing `security-alerts` channel, respond and react in an incident thread, message the new group, rename it, and return its ID.

**Result:** Failed because the agent confused two entities with the same name.

The seeded workspace contains the public channel `security-alerts` (`C005`). The agent also created a group chat named `security-alerts` (`G003`). It sent `Incident analysis started` to `G003` with `send_group_message` instead of posting it to `C005` with `post_message`.

All remaining actions were correct: the agent replied to the incident thread, added the heart reaction, welcomed the group members, renamed the group to `Design Ops Sync`, and returned `G003`.

- **Failed checks:** channel message state and required posting trajectory
- **Primary cause:** failure to distinguish entity types during a name collision

## Task Manager environment

### Task 1: Conditional bulk reassignment

The agent must reassign every task owned by Jordan Kim (`U004`). Tasks with a milestone go to Morgan Patel (`U002`). Tasks without a milestone go to Riley Stone (`U003`) and retain their existing labels while adding `needs-triage`. The final answer must list every modified task ID.

**Result:** 11 of 12 checks passed.

A single `list_tasks` call returned all five relevant tasks. The agent correctly updated `TASK006`, `TASK008`, and `TASK032` to Morgan Patel. It also correctly updated milestone-free `TASK031` to Riley Stone and appended `needs-triage` without replacing its existing labels.

The agent skipped `TASK009`, even though it appeared in the list response, and omitted it from the final comma-separated answer.

- **Completed:** `TASK006`, `TASK008`, `TASK031`, `TASK032`
- **Missed:** `TASK009`
- **Primary cause:** incomplete processing of a known result set
