# Slack incident reconciliation run report

## Task idea

This task puts an agent in a simulated company Slack workspace and asks it to own the final readiness review for the Acme migration. The agent is Ben Ortiz, a platform engineer. The episode starts on Wednesday, August 19, 2026 at 3:00 PM Pacific Time. Time comes from a deterministic virtual clock.

The seeded workspace has 51 users, 24 channels, direct messages, group chats, and 1,690 visible messages. Five Acme channels contain more than 200 messages apiece. Useful evidence sits deep enough in the history that reading only the newest four 50-message pages is insufficient. Search is a valid alternative when the agent finds a useful incident ID, policy name, or API name.

Daniel Cho asks Ben to settle four areas: SSO, the historical export, EU workspace access, and the migration schedule. Ben also has to review the open Acme changes in `#debugging`. The request is intentionally incomplete. Cutover coordination has moved to a public channel Ben has not joined, and that channel reveals rollback, bridge coverage, and the final rehearsal as additional dependencies.

The workspace changes in response to Ben's work. Reviewing a bad SAML helper releases a corrected revision. Approving that revision releases CI and deployment results, followed by a new EU-2 signature failure. The export review starts a March backfill, whose 26-record difference can only be explained with an older exclusion policy. The permissions checker first looks green, then proves unreliable, then exposes a missing EU-Legal entitlement after it is fixed. A rehearsal later invalidates a provisional rollback result and reopens work that appeared complete.

There are 20 deterministic scenario events and eight required review decisions. At the terminal state, the export, permissions, rollback, and rehearsal are complete. SSO is still blocked because EU-2 cannot verify the `acme-eu-2025` signing key. The correct final decision is `BLOCKED` for that reason alone.

## Why the task is fair

Every required fact is reachable through the Slack tools. Channel history is paginated in 50-message pages, threads open separately, and search returns bounded results with message and thread identifiers. The old evidence is difficult to find, but recent conversation gives the agent search terms such as `IDP-ACME-014`, `ACME-DRYRUN-07`, `ACME-ACCESS-04`, and `DIR-PAGE-311`. A careful agent can use search or ordinary pagination.

Future messages do not exist in the live message table until their event fires. Before release they cannot appear in history, search, thread replies, unread counts, reactions, message totals, or thread metadata. The tool call that triggers an event returns before the new messages become visible. This makes a second read necessary and prevents the agent from seeing the answer early.

The event rules accept more than one reasonable way to ask for information. A clarification may be posted in the relevant thread, in the working channel, or in a direct message to the person who knows the answer. Keyword matching uses stems and groups of required ideas rather than one memorized sentence. Tests cover plausible accepted wording and near misses that should stay rejected. The only exact phrase is `Looks good to me`, which Daniel supplies in the task instruction for approvals.

The agent container has the Slack client and MCP server. A sidecar owns the simulator and SQLite database. The agent socket exposes normal Slack operations; a separate admin socket is reserved for reset and verifier export. The environment runs without public networking. Only the model request made by the agent harness is allowlisted.

The fixture also includes correct code that looks suspicious, defective code whose problem depends on another channel, closed review threads, stale pins, saved items, and unrelated notifications. This keeps simple shortcuts from working. An agent cannot assume every review is wrong, treat followed threads as its task list, or accept a checkmark as current evidence.

Two independent correct trajectories exercise the full path: the oracle and `competent_run`. They use different review orders and evidence routes, and both score 1.0. The tests also verify that every check missed by the short capability-failure episode is achievable in the competent episode.

## Model runs

The `jobs` directory currently contains three recorded Claude Opus 4.7 runs. Harbor completed all three normally, marked each episode valid, and recorded no environment exception. The rewards below are fresh evaluations of each saved state export under the current consequence-weighted verifier; the canonical `verifier/reward.json` files have been updated to these values.

| Trial                                    | Task checksum |   Reward | Success | Output tokens |  Cost |
| ---------------------------------------- | ------------- | -------: | ------: | ------------: | ----: |
| `slack-incident-reconciliation__x2mZpu3` | `bfd22a4d...` | 0.292084 |       0 |        18,263 | $6.06 |
| `slack-incident-reconciliation__FCHFx5c` | `bfd22a4d...` | 0.549779 |       0 |        30,229 | $9.23 |
| `slack-incident-reconciliation__LQUtkpV` | `b5a927cd...` | 0.276464 |       0 |        26,459 | $7.17 |

`x2mZpu3` completed a first pass over the reviews and posted an intermediate `BLOCKED` assessment, but it did not re-read after its writes. It therefore missed the revised SAML and permissions reviews, the deployment and changed SSO failure, export reconciliation and reopen, permissions verification, rollback, rehearsal, and bridge-coverage transitions. Its score consists of useful initial-state work and the timing confirmation, minus four small stale-evidence charges.

`FCHFx5c` made the most progress. It correctly drove the SAML review through revision and deployment, surfaced the new EU-2 key failure, found the export policy and reconciliation, and confirmed the migration time. It nevertheless approved the defective `provisionAllUsers` implementation, failed to give the exact approval required for the corrected permissions revision, and stopped before permissions completion, the reopened export review, rollback, rehearsal, and bridge coverage. A partial score near 0.55 is a reasonable measure of its substantial but incomplete progress. Whether knowingly approving defective migration-critical code should instead be configured as a forbidden-action veto is a policy choice; the current contract treats this particular review error as a heavily weighted missed decision, not a disqualifying act.

`LQUtkpV` correctly identified the initial SAML and permissions defects and the cross-thread duplicate-provisioning risk. It then posted a T0 `BLOCKED` report and stopped before the dynamic follow-up chain. Its previous stored zero was not fair because it came from treating cited stale evidence as a fatal failure even when the answer used that evidence to explain why an older conclusion was unreliable. The current verifier gives ordinary partial credit and applies `0.026923` in proportional stale-evidence charges, producing `0.276464`. That is materially fairer, although the lexical stale-evidence detector still slightly under-credits this trajectory.

The common failure is an agent-capability failure as each agent formed an initial picture, acted once, and stopped instead of repeating the required act-read-replan loop. All three trajectories are valid, have no tool failures or task exceptions, and preserve complete state exports. The deterministic oracle and the independently ordered `competent_run` both reach the terminal state and score 1.0, so a human following the available references, threads, and post-action updates has a complete solution path.

## Verifier design

The graded preset is `full_layered_deterministic`. Three structural layers contribute positive credit:

- final state checks task outcomes such as review decisions, Daniel's answer, evidence citations, owners, and final verdict;
- milestones checks which events fired and whether the terminal answer followed the required evidence;
- trajectory checks whether the run made task progress and subtracts 0.02 for each individually identified off-task action;
- negative and causal-order checks are zero-weight vetoes. Any forbidden action or impossible causal ordering makes the entire reward zero. Heuristic audit findings are diagnostic only and never change reward.

The configured weights are 0.40 for final state, 0.50 for milestones, and 0.10 for trajectory. Requirements within those layers are also weighted by consequence. Final SSO diagnosis and rehearsal completion are worth more than routine review bookkeeping, for example.

The verifier does not require a particular investigation recipe. Search, pagination, a direct question to an owner, and a thread-based question receive the same credit when they produce the same authoritative state. Causal ordering is checked only to reject impossible trajectories, not to award points for following one preferred path.

The pass threshold is 0.999. Partial credit is still useful for comparing failed runs, but it cannot turn an incomplete episode into a pass. The final answer must be posted after the four terminal events: SSO key investigation, export reopen closure, permissions verification, and rehearsal completion.

Grading reads the state exported by the Slack sidecar and the action log stored by the simulator, and nothing else. It does not trust files written in `/logs/agent`, and there is no longer a `migration_readiness.json` scorer beside it: the layered reward is the whole key stream.

Before grading a run, Harbor executes 16 self-test suites covering the clock, Slack behavior, MCP transport, scenario activation, trigger wording, verifier arithmetic, reward validation, integrity checks, fixture shape, and RL contract. A self-test failure sets `valid=0`, which separates an environment problem from a model failure. All self-tests passed in the three saved runs, and each run has `valid=1` with no recorded exception.

## Limitations

The scenario is deterministic. This makes failures reproducible, though it does not model the timing and wording variation of a live team. Other users' future messages are prewritten rather than generated in response to the substance of a free-form conversation.

Some gates still depend on lexical matching. The accepted phrase tests cover known reasonable variants, but an equally good sentence may use vocabulary outside those cases. Exact approval matching is defensible because the requested text is supplied verbatim, though it can penalize an otherwise sound approval that adds a qualification.

The workspace has broad Slack behavior but no files or attachments. Code is reviewed from message text; the task does not compile or execute the snippets. The simulator can therefore test review judgment and state tracking, not repository navigation or build debugging.

The reactive scenario is represented as data, but the Acme message corpus still lives in `seed.py`. Adding a new scenario would require separating more of the fixture from the general Slack environment.

The model evidence is small. There are three Opus 4.7 runs, and the task checksum changed between the second and third. The table supports a qualitative failure analysis, not a statistically reliable estimate of success rate or a clean before-and-after comparison. No other model family has been run on the same current checksum.

The superseded-evidence check is still lexical. In `LQUtkpV`, it charges some old message citations even though the agent cited them while rejecting their conclusions. The charges are now proportional rather than fatal, but distinguishing reliance from explicit repudiation would make this part of the score more precise.

The reward only checks task facts encoded in the contract. It does not attempt to assess subjective attributes of the final explanation.

Three specific gaps are known and unrepaired. Nine fields in `GROUND_TRUTH`, among them `start_time`, `duration_minutes`, and `expected_downtime_minutes`, are read by no check; the contract confirms that a report cites the message settling the schedule without confirming that it states the right window. `AREA_TERMS["historical_export"]` contains `partition`, which is a code-review term belonging to `buildPartitionWindow` rather than to the export area, so a report discussing that helper could be credited with coverage it did not earn — this changes no recorded score, but it is a latent false positive. And `search_messages` applies no byte ceiling: a broad query still returns roughly 100 KB, which the harness may drop in full rather than truncate, leaving the agent with nothing where a partial result would have served.

## Reproduction commands

Run these from the repository root. Docker must be running, and Harbor must be installed.

The runs below are Claude Opus 4.7 through OpenRouter. Put `OPEN_ROUTER_KEY` in `.env` at the repository root and name that harness explicitly, because the wrapper now defaults to the task's own adapter agent on a local Ollama model:

```bash
AGENT=claude-code MODEL=anthropic/claude-opus-4.7 ./example_tasks/slack/run.sh
```

The wrapper clears earlier task containers, uses high reasoning effort, and writes the job under `example_tasks/slack/jobs/harbor`. Plain `./example_tasks/slack/run.sh` runs the default local model instead and needs no key at all; it will not reproduce the scores in this report.

To stop a run or clear old task containers without starting another one:

```bash
./example_tasks/slack/kill.sh
```

To run the deterministic reference solution:

```bash
harbor run \
  -p ./example_tasks/slack \
  -a oracle \
  --force-build \
  --yes
```

To run the same verifier self-tests locally:

```bash
cd example_tasks/slack
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=environment:tests:.

# The list is read out of test.sh rather than written down again, so it cannot
# drift from the one the verifier actually runs -- which it had, by a suite.
sed -n 's/^for suite in \(.*\); do$/\1/p' tests/test.sh | tr ' ' '\n' |
  while read -r suite; do python3 "tests/${suite}.py" || exit 1; done
```

To grade a collected sidecar export again:

```bash
cd example_tasks/slack
PYTHONPATH=environment:. python3 -m verifiers.run \
  --state jobs/harbor/<job>/<trial>/artifacts/var/lib/slack/state-export.json
```

The saved run's own `result.json`, `verifier/details.json`, `verifier/reward.json`, and `verifier/test-stdout.txt` contain the score, individual checks, validity flag, and environment test results.
