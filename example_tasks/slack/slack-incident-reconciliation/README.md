# Slack Simulation Environment

A self-contained, reusable Slack simulator for building Harbor and RL tasks
that evaluate long-horizon state tracking, replanning, evidence retrieval, and
decision-making in a realistic workspace. The environment is not tied to one
company, project, or readiness workflow.

This repository includes one incident-reconciliation scenario as a concrete
fixture and end-to-end example. Its people, channels, messages, transitions,
instruction, and verifier contract are scenario data layered on top of the
generic Slack runtime. Other tasks can use the same tools, state model,
pagination, access controls, virtual clock, lifecycle, and transition engine
with their own seed and grading contract.

## Reusable environment and bundled scenario

| layer | reusable across tasks | supplied by a scenario |
| --- | --- | --- |
| Slack runtime | users, conversations, messages, threads, reactions, search, pagination, read state, permissions, and virtual time | identities, memberships, message corpus, and starting clock |
| Dynamic behavior | generic rule evaluation and latent-state activation | event definitions, triggers, dependencies, and future messages |
| Agent interface | MCP and CLI tool schemas, session lifecycle, and bounded observations | system/user prompts and task instruction |
| Evaluation | deterministic verifier primitives, weighted layers, action tracking, and veto handling | required outcomes, evidence, ordering, weights, and forbidden actions |

The sections labelled **bundled scenario** document the fixture currently used
to exercise the simulator. They are examples of what can be authored with the
environment, not restrictions built into the Slack surface.

## Runtime architecture

```text
agent in main container
  -> slack MCP server (stdio)   registered by Harbor from task.toml
  -> Unix-socket Slack simulator API in sidecar
  -> authoritative SQLite workspace
```

The `main` image contains the agent and client only. The `slack` sidecar owns
the simulator, seed, transition engine, and database. The model-facing socket
exposes Slack tools; a separate restricted admin socket supports lifecycle and
verifier export. The agent cannot read the database, seed, latent events, or
verifier ground truth.

### The agent is handed the workspace, not told about it

`task.toml` declares the workspace as an MCP server:

```toml
[[environment.mcp_servers]]
name = "slack"
transport = "stdio"
command = "/usr/local/bin/slack-mcp"
```

Harbor registers that with whichever agent runs the task -- user-scoped for
Claude Code, `config.toml` for Codex, and so on -- so all 48 Slack tools arrive
in the agent's own tool list, each carrying its JSON schema, before the first
turn. A scenario's `instruction.md` can therefore contain only the simulated
request: it does not need to name commands, demonstrate verbs, or describe the
client. Finding information in the workspace can be part of a task;
discovering that the Slack interface exists is not.

`slack_sim/mcp_server.py` speaks newline-delimited JSON-RPC 2.0 on stdio
(`initialize`, `tools/list`, `tools/call`, `ping`) and proxies every call to
`agent.sock`. Its tool list is generated from `tool_definitions.py`, so the
agent's surface cannot drift from the workspace's. It is a client: it imports
only `protocol` and `tool_definitions`, and a test asserts that, because this
module ships in the agent's image where the seed and service must never be.
A workspace refusal is returned as the workspace's own envelope with
`isError: true`, so a denial stays legible instead of arriving as an empty
success.

The `slack` CLI remains in the image for the oracle solution and the test
suites, and as a fallback for harnesses that do not speak MCP.

### Two loops over one simulator

The same workspace serves an evaluation loop and a training loop. They differ
in who drives the turn and what the reward is for; the world underneath is
byte-identical, which is the point -- an RL policy trains against the thing it
will be graded on.

```text
EVALUATION  (harbor run)                  TRAINING  (rl_env.py / in-process)
------------------------                  ----------------------------------

  main container                            trainer process
    agent: Claude Code, Codex, oracle         |
      |                                       |  reset / step / state
      |  MCP over stdio, registered           |
      |  by Harbor from task.toml             |
      v                                       v
    slack-mcp                               SlackIncidentEnvironment
      |                                       |   environment.py, contract 7.0
 =====|========= container boundary ====      |
      v                                       |
  slack container                             |
    agent.sock  0666                          |
      |                                       |
      v                                       |
    server.py ------> execute_tool <----------'   direct call, no socket
                           |
              .------------+------------.
              v            v            v
         service.py   scenario.py   tracker.py
         48 tools     20 gated      what the agent
         virtual      events        actually observed
         clock
                           |
                           v
                    SQLite workspace


AFTER THE AGENT PHASE
---------------------

  admin.sock  0600 root --> export_state --> state-export.json
                                                   |
                                                   v
                                        verifiers/  four weighted layers
                                                   |
                                                   v
                                       /logs/verifier/reward.json
```

**Evaluation.** Harbor reads `task.toml`, starts both containers, and registers
`slack-mcp` with whatever agent runs the task. The agent never learns a command
name: the 48 tools arrive in its own tool list. After the agent phase Harbor
runs `tests/test.sh` as root, which exports world state over the privileged
socket and grades it with `verifiers/`. Reward is the four-layer contract --
final state 0.40, milestones 0.50, trajectory 0.10, negative 0.00 as a veto --
and is written once, at the end.

**Training.** `SlackIncidentEnvironment` skips the socket entirely and calls
`execute_tool` in-process, so a rollout costs no IPC and no containers. Every
step returns a shaped scalar: `workspace_progress` scores 16 observable
milestones, subtracts the side-effect penalty, and the step's reward is the
*delta* in that progress. A rollout terminates when all 16 are met with no
penalty, and truncates at `max_turns` (default 100).

The split in what each loop rewards is deliberate. Shaping counts only what the
world can see -- which events fired, whether an answer was posted at all --
because the verdict and the owners are short strings a policy could assert
without reading anything. Whether the answer is *right* is the verifier's call,
after the episode. Rewarding it per-step would put the shaped signal in direct
opposition to the graded one.

## Bundled scenario

### Initial state

Claude acts as Ben Ortiz, a platform engineer. The virtual clock begins at
Wednesday, August 19, 2026, 3:00 PM PT. The deterministic seed contains:

- 50 users with profiles and reporting relationships;
- 23 public and private channels, plus DMs and group chats;
- 1,679 initially visible historical messages, with at least 200 top-level
  messages in every Acme-relevant channel;
- threads, reactions, mentions, read cursors, unread state, memberships, and
  private-channel access control.

The T0 migration state is blocked: EU authentication is failing, March is not
yet reconciled, and EU-Legal permissions have no trustworthy end-to-end
verification. A later 9:30 PM question makes the previous 9:00 PM approval
ambiguous. The final rehearsal has not run, and rollback has not yet surfaced
as a prominent requirement. Same-day 4:30, 5:30, 6:30, 7:30, and 8:30 gates
create prioritization pressure.

### Deep history

Four required historical discussions sit more than 200 top-level messages
behind the newest activity in their channels:

- `IDP-ACME-014`: canonical SAML audience, current/previous signing keys, and
  regional metadata-cache invariants;
- `ACME-DRYRUN-07`: exactly 26 synthetic load-test records are excluded;
- `ACME-ACCESS-04`: all entitlements plus authentication, both workspace
  opens, restricted-area access, and the Legal export action are required;
- `DIR-PAGE-311`: the upstream API can repeat a page-boundary record and the
  downstream provisioning operation is non-idempotent.

Recent conversations naturally refer to these identifiers. The agent can
reach the old evidence through normal 50-message pagination or Slack search.
Channel history returns roots only; thread content requires
`get_thread_replies`.

## Dynamic scenario engine

### The environment hosts a task; it does not contain one

`slack_sim` is a Slack workspace and nothing more: channels, threads,
membership, search, a clock, and tools that mutate the database the way Slack
does. Project-specific behavior is supplied by scenario data.

Everything task-shaped is a `Scenario` (`scenario.py`), which is data:

| field | meaning |
| --- | --- |
| `events` | the things that can happen in this world, once each |
| `latent_notifications` | inbox entries that arrive with an event |
| `latent_memberships` | people who join a channel when an event fires |
| `latent_messages` | what other people say when one happens |
| `rules` | what the agent has to do for one to happen |

`tracker.py` is a generic evaluator over those rules. Every completed tool
call passes through it, so it is also the one place that observes the agent's
whole action stream. It supports five
trigger kinds — `reply_keywords`, `reply_exact`, `observed`, `channel_joined`
and `all_of` (a pure dependency closure) — each gated by `requires_activated` /
`requires_pending`, which is how a scenario expresses ordering. Evaluation
runs to a fixed point, so a closure resolves in the same call that completed
its last prerequisite. A second task means writing a second `Scenario`, not
editing the environment; `tests/test_scenario_engine.py` exercises the engine
with independent scenarios invented in the test file.

The bundled fixture's rules live in `seed.py` as `SCENARIO_RULES`, and its
workspace corpus is still module-global. `Scenario` therefore carries the
reactive half of a task today, while a new fixture also supplies seed data.
Moving each corpus into its own scenario package is the remaining step toward
drop-in scenario selection.

### Gates reject wrong reasoning, not unlucky wording

A gate exists to check that the agent demonstrated something -- that it noticed
a contradiction, asked the person who knows, or read the code closely enough to
see the defect. Rejecting an agent that did those things measures nothing, so
`reply_keywords` rules declare *where they listen* rather than assuming one
thread:

| field | meaning |
| --- | --- |
| `thread_id` | a reply in this thread, the canonical place |
| `channel_id` | any message the actor posts in this channel |
| `recipient_ids` | a direct message to any of these people |
| `tools` | which operations can carry it (default `reply_to_thread`) |

Reaching any one address is enough; the keyword groups still decide whether the
agent showed the understanding. The groups hold stems rather than inflections,
so `approval`, `approved` and `approving` are the same evidence. Scenario tests
should cover equivalent wording and legitimate placement choices so a gate
measures the intended decision rather than one exact phrasing or route.

`Looks good to me` stays an exact match. The task instructs that string
verbatim, so widening it would measure something nobody asked for and make an
accidental approval indistinguishable from a deliberate one.

Widening *where* exposed ordering that had been implicit in *where*: a rule
anchored to a thread cannot fire before that thread exists, so two events never
had to declare the prerequisite that made their replies possible. Both now do,
and `test_event_activation.py` fails the build if any latent reply can be
published before its parent. The tracker is also defensive at runtime -- an
event whose parents are not visible stays pending instead of raising, because a
scenario-authoring mistake must never turn a valid tool call into an error the
agent has to interpret.

`tests/test_trigger_fairness.py` holds the accepted phrasings and placements,
alongside the mentions that must still be refused: a gate that fires on any
mention of its topic is not a gate.

### Bundled scenario: lived-in workspace design

The workspace is a company Slack that happens to be running an Acme migration,
not a migration fixture with some decoration. 51 users, 24 channels, 1,690
visible messages at T0, plus DMs, group chats, reactions, private channels and
two weeks of unrelated history.

The Acme work spans five channels Ben is in — `#acme-migration`,
`#identity-eng`, `#data-ops`, `#enterprise-support`, `#debugging` — and one he
is **not**: `#acme-cutover-bridge`. That channel is public and holds the
same-day work (rehearsal, rollback, staffing, go/no-go), and a message Ben can
see mentions that coordination moved there. Nothing tells him to join. The
rollback and rehearsal threads live inside it, so the cutover-operations half
of the task is unreachable until he finds it.

Side state is lived-in rather than authoritative, which is the point of
including it:

| | at reset | why it misleads |
| --- | --- | --- |
| pins | 3 in `#acme-migration` | the 8:00 PM kickoff is still pinned and wrong; the 90-minute window is still pinned and right |
| saved items | 5, Ben's own | the July dry run and the pagination incident matter later; the kickoff checklist is dead |
| followed threads | 4 | none of them is an open review, so `list_followed_threads` is not a task list |
| notifications | 12 for Ben | mentions, DMs, one thread reply, one reaction — most of it unrelated |
| reactions | ✔ on stale claims | Daniel's "SSO complete" still carries checkmarks |

Ben also wrote one of the stale messages himself ("Sounds like 9:30 PM is
likely for Thursday"), posted while the schedule genuinely was ambiguous. He
can delete or correct it once 9:00 PM is reconfirmed; nothing requires it.

Deep history sits more than 200 top-level messages back in its channel and is
reachable by search or pagination: the first Acme certificate rotation and its
signing-key selection rules, the July dry-run exclusion of exactly 26 synthetic
records, the original EU-Legal acceptance criteria, and the directory-pagination
and provisioning-idempotence incidents. Recent messages point at them by name
("the first Acme cert rotation", "the dry-run exclusion policy") without
restating them, so the agent has to build the query.

### Every episode starts identical

`seed_database` drops the workspace and rebuilds it, and it is the only path
to a workspace: `serve()` calls it on startup and `reset()` calls it per
episode. An episode never inherits another episode's messages or half-fired
events. Seeding costs about 90 ms, and reseeding twice produces byte-identical
snapshots.

### Latency of release

Future Slack messages are stored in a separate `latent_messages` table. They
are absent from the live `messages` table until an event fires, so they cannot
affect history, search, thread metadata, unread counts, reaction metadata, or
any other model-facing field. The transition evaluator considers only the
serialized tool result actually returned to the agent—not rows a search might
scan internally. It also runs after a call has produced its result, so the
call that satisfies a rule never sees the messages it released.

The bundled scenario contains 20 state and observation events. Its main
sequence is:

1. A correct critique of `audienceMatches` releases a corrected revision.
2. Approval releases CI, deployment, and a four-account smoke test. Combining
   the new EU-2 failure with old rotation evidence releases the regional
   metadata/key diagnosis.
3. Approval of `buildPartitionWindow` releases the March backfill and a
   26-record discrepancy.
4. Observing the historical exclusion policy releases exact reconciliation
   and reopens the export review.
5. Approval of the reopened export revision closes it.
6. A correct critique of `hasRequiredAccess` releases an all-match revision.
7. Approval releases the checker failure and remediation, followed by a
   deliberately insufficient “workspace loads” test. Only a later restricted
   export-workflow test completes EU access.
8. Ben must ask whether 9:30 was a real change. Nina confirms that it was only
   a question and that 9:00 PM remains authoritative.
9. That clarification reveals rollback as an omitted dependency and records a
   bridge-coverage change. Sam's initial invocation is only provisional.
10. The virtual clock advances deterministically to the 7:30 PM rehearsal.
    Component results invalidate rollback because of a stale config pin. Ben
    must request a correction and rerun before rehearsal and rollback close.

The terminal state remains `BLOCKED`, but only because of the new EU-2
signature/key failure. Historical export and EU permissions become verified
complete. Full credit is impossible from T0 evidence or from a report posted
before the terminal events.

### Investigating a failed episode

When an episode ends with the workspace unchanged, four different causes look
identical from the outside, and they call for opposite fixes:

1. the agent never took an action a rule could consider;
2. it took a reasonable action a rule rejected;
3. a rule matched and the event fired, but nothing became observable;
4. propagation broke somewhere else.

The tracker records why each rule matched or did not, one predicate at
a time, with the candidate keywords and what actually matched. Set
`SLACK_EVENT_TRACE=1` on the `slack` service to have those blocks written to its
stderr, or pass an `EventTrace` to `execute_tool`. Either way the trace is
world-side: it is never attached to a tool result and never crosses
`agent.sock`.

```text
[event-eval]
event_id=timing_confirmed
rule_id=r040_timing_confirmed
thread=CUT001
raw_text="Is 9:30 still the plan?"
predicates:
  thread_parent=CUT001: PASS
  keyword_group_1:  candidates=["9:30","930","nine thirty"]  matched=["9:30"]  PASS
  keyword_group_2:  candidates=["confirm","question","request","approved","actual"]
                    matched=[]  FAIL
final_match=FAIL
```

`tests/event_audit.py` runs this by hand, against a phrasing or against a real
Harbor trajectory:

```bash
PYTHONPATH=environment:tests python3 tests/event_audit.py \
    --event timing_confirmed --body "Is 9:30 still the plan?"

PYTHONPATH=environment:tests python3 tests/event_audit.py \
    --event timing_confirmed --trajectory ../jobs/harbor/<job>/<trial>/agent/trajectory.json
```

It prints the state before, every predicate with its candidates, whether the
event moved, and what became observable afterwards. `tests/event_diagnostics.py`
holds the reusable pieces: a `Workspace` that drives the world through
`execute_tool`, and a `probe` that asks history, threads, search, notifications
and thread metadata whether a message is reachable -- so "activated" is never
mistaken for "the agent could see it".

Two asymmetries matter when reading these results, because both look like bugs
and are not:

* `get_channel_messages` returns top-level messages only, so a released *reply*
  is absent from channel history exactly as the agent's own replies are;
* search covers public channels the actor has not joined, while reading their
  history does not -- which is how `#acme-cutover-bridge` is meant to be found.

`tests/test_event_activation.py` characterises all of this, including a matrix
of ten plausible clarification phrasings. Its expected-fail rows are findings
about the current lexical gate, not endorsements of it.

## Bundled scenario review work

Seven plausible `[review-needed]` threads exist at T0. Two are already closed
and five are open. Those five require eight review decisions because SAML and
permissions receive corrected revisions and export later reopens. The mix
includes defective code, correct-but-suspicious code, and a cross-thread defect
that is not obvious from the local code alone.

## Operation coverage

48 tools. Everything a reader would expect of a Slack workspace except files
and attachments, which are deliberately out of scope: message bodies are plain
text and there is no upload surface.

| Domain | Operations |
| --- | --- |
| Messages | `post_message`, `reply_to_thread`, `edit_message`, `delete_message`, `get_channel_messages` |
| Threads | `reply_to_thread`, `get_thread_replies`, `follow_thread`, `unfollow_thread`, `list_followed_threads` |
| Search | `search_messages`, `search_users`, `search_channels` |
| Reactions | `add_reaction`, `remove_reaction`; grouped inline on every message |
| Channels | `create_channel`, `join_channel`, `leave_channel`, `archive_channel`, `edit_channel_name` |
| Private channels | `create_channel(is_private)`, `invite_to_channel`, `remove_from_channel` |
| DMs | `create_dm_message`, `send_dm_message`, read via `get_channel_messages` |
| Group DMs | `create_group`, `send_group_message`, `add_group_chat_participants`, `edit_group_chat_name` |
| Users | `list_users`, `search_users`, `set_status`, `set_presence`, `edit_display_name` |
| Membership | `invite_to_channel`, `remove_from_channel`, `list_channel_members` |
| Pins / saved | `pin_message`, `unpin_message`, `list_pins`, `save_message`, `unsave_message`, `list_saved_items` |
| Permissions | workspace role, channel owner, membership role — enforced on archive, remove, `@everyone`, cross-user status |
| Unreads | `mark_conversation_read`, per-message `is_unread`, per-conversation `unread_count` |
| Mentions | `tag_people`, `tag_here`, `tag_everyone`, `tag_user_group` |
| Notifications | `list_notifications`, `mark_notification_read` |
| Workspace metadata | `list_channels`, `list_users`, `list_user_groups`, `list_chats`, topics, teams, roles |

Semantics worth knowing:

- **Delete really deletes.** The row goes, along with its reactions, mentions,
  pins, saved items and notifications. There is no tombstone and no hidden
  copy, so a verifier checking whether a deletion happened diffs the workspace
  against the seed rather than trusting a flag the environment set about
  itself. Authors delete their own; workspace admins delete any. Deleting a
  thread root deletes the thread, replies included, the way Slack does — the
  result reports every id that went in `deleted_message_ids`. A surviving
  reply that answered a deleted message falls back to answering its thread.
- **Pins are shared, saved items are private.** A pin belongs to the
  conversation and everyone in it sees the same list; saved items are visible
  only to the person who saved them.
- **Replying to a thread follows it**, the way Slack does, so later replies
  notify you without a separate step.
- **Notifications are part of the initial state.** `SLACK_NOTIFICATIONS` is
  declared seed data like reactions or read cursors, built from the fixture at
  import time so it cannot drift when the history changes, and inserted
  verbatim at seed time — the service computes nothing. Every run therefore
  starts from the same 58-row inbox. At runtime, new messages notify the same
  way: mentions, direct messages and replies in followed threads, never your
  own messages.
- **Archiving keeps history and refuses writes.** The gate sits on the shared
  membership check, so every writing path inherits it.
- **Presence** is `active` / `away`; it is not inferred from activity.

### Search grammar

`search_messages` accepts phrases and filters:

```
"automated access checker"        exact phrase, in order
from:priya                        by sender name or handle
in:identity-eng                   by channel or group-chat name
before:2026-08-05                 strictly earlier
after:2026-08-05                  on or later, so before/after partition
on:2026-08-19                     one workspace-local day
```

Filters combine, and every one must match. A filter naming nobody or no
conversation raises `invalid_search_filter` rather than returning nothing
silently — a typo should not look like an answer. Visibility is applied after
filtering, so `in:` on a private channel you are not in returns zero results
instead of confirming the channel exists. Dates resolve against a pinned
Pacific offset, so they do not depend on a tz database in the image.

## Slack observation contract

- `list_channels` and `list_users` are cursor-paginated.
- `get_channel_messages` returns at most 50 newest-first top-level messages and
  an opaque, stable cursor for the next older page.
- `get_thread_replies` explicitly opens a complete thread.
- `search_messages` returns bounded matches with identifiers and at most one
  neighboring message on either side; it does not expand conversations.
- Mutations persist for the episode.
- A refusal carries a `code` and a `type`, and the two say different things: a
  `ToolError` code means the rules declined the request and asking differently
  may work, while `storage_error` means nothing the agent asks will work.
  `internal_error` is now the residue rather than the default for everything
  unexpected.
- **Absent means absent.** A field whose emptiness carries no information is
  left out rather than sent empty: no `thread_ts` means no thread, no
  `reactions` means nobody reacted. Fields whose falsy value answers a question
  always ship -- `is_member: false`, `unread_count: 0`, a mention's
  `ordinal: 0`. Results are serialized compactly, since nothing reads them but
  a parser. Together these roughly halve what a sweep of the workspace costs in
  context, which is the resource this task actually contends for.

All timestamps come from SQLite-backed virtual time. Ordinary reads do not
move time. Agent writes and Slack-side event publications advance it
deterministically; no host wall clock is consulted.

## Verifiers

`verifiers/` is a task-agnostic grading stack. Five verifier types compose into
one auditable verdict:

| type | asks |
| --- | --- |
| `ExactStateVerifier` | did specific state fields end up with specific values |
| `PolicyVerifier` | does a Python expression over the end state hold |
| `EventVerifier` | did the required events appear at all |
| `TemporalVerifier` | did they appear in an order the story permits |
| `NegativeVerifier` | did anything forbidden happen |

`LayeredVerifier` composes them into four layers -- final state, milestones,
trajectory, and negative -- and applies one rule that the arithmetic cannot
overturn: **any forbidden side effect or impossible ordering makes the reward
zero.** Partial credit describes how much legitimate work was done; it never
turns a prohibited route into a price the agent can choose to pay.

**This is the reward Harbor stores.** `tests/test_migration_readiness.py`
grades every episode with `TieredRewardEngine` under the preset named in
`verifiers/experiment.yaml` (or `TASK_REWARD_PRESET`), and writes the layered
number as `reward`. It is the only grader: the key stream is the layered reward
and nothing else, because two answers to "how did this run do" is one too many.

`TieredRewardEngine` turns that into a `RewardBreakdown` where every term is
readable: each layer's weight, score and contribution, then any penalties, then
the total. The reward is fully deterministic and requires no grading model.

### Partial credit, and what it is allowed to pay for

Checks carry task-authored weights. Final SSO diagnosis, verified permissions,
the reopened export review, and rehearsal completion are worth more than
routine bookkeeping. The weights reflect operational consequence.

Two rules keep that from paying for the wrong thing:

**Abstention earns nothing.** An untouched workspace scores exactly `0.0`. The
The `trajectory` layer asks only whether the episode made real task progress.
It does not require a fixed number of searches, thread reads, or one preferred
route to evidence. Search and pagination receive equal credit when they expose
the same state. Each off-task action subtracts separately. The `negative` layer
carries weight `0.00` because its job is to veto rather than to add.

**Credit is for what you reported; the veto is on what you stood behind.**
Evidence citations are read across every message in the request thread, because
a long update gets split and crediting only the last message would price a
Slack habit. The standing verdict, and the stale-evidence veto, read the last
message alone -- and that veto fires only on a stale claim the report *rests*
on. Naming one in order to reject it is the behaviour this scenario is built to
reward, so a sentence marking a claim as superseded is not an offence.

The scale that results, anchored at both ends:

| episode | reward |
| --- | --- |
| oracle solution | 1.000 |
| Opus 4.7, `6osSdvQ` | 0.274 |
| Opus 4.7, `pibbWtp` | 0.260 |
| agent that did nothing | 0.000 |

Under the legacy reward the oracle itself topped out at 0.42, because two
thirds of the weight ran through a `/app/migration_readiness.json` artifact the
prompt never names. The graded answer is the reply in the thread Daniel asked
in -- what the task actually asks for. That scorer and the artifact are gone:
grading takes one argument, the world's state export.

### Reward presets

`reward_preset` in `verifiers/experiment.yaml`; `VerifierComposer` and
`TieredRewardEngine` resolve the same name, so a run's output always states how
it was graded.

The graded default is `full_layered_deterministic`: the same episode produces
the same number every time, on any machine, offline. Final state carries 0.40,
milestones 0.50, and trajectory efficiency 0.10.

| preset | behaviour |
| --- | --- |
| `full_layered_deterministic` | **the graded default.** All four deterministic layers |
| `full_layered_deterministic` | deterministic weighted partial credit with audit diagnostics |
| `binary_final_state` | final-state checks plus forbidden-action veto; reward is exactly 0 or 1 |

### Reward-hacking audit

`RewardHackingAuditor` is a separate diagnostic pass that flags skipped
milestones, suspiciously short episodes, one thread hammered repeatedly, and
episodes that wrote and never read again. `RewardHackingAuditor.for_verifier(...)` inherits the
milestone list from the `LayeredVerifier` that will issue the pass, so the two
cannot drift apart. Its findings are diagnostic and do not change reward.

### Does the reward mean what it says?

`tests/test_reward_validation.py` grades two realistic episodes end to end
through the same entry point Harbor calls. `tests/reward_episodes.py` holds
them.

**`competent_run`** is a second correct solution, written independently of the
oracle: reviews in a different order, evidence reached by different routes, the
timing question asked of Nina by DM rather than in-thread, the final update
split across two messages. It scores **1.000**. A reward that only pays the
script it was written against is measuring the script.

**`capability_failure_run`** is modelled on the recorded Opus 4.7 episodes:
read the obvious channels, approve the reviews that look fine, never discover
that coordination moved to a channel it was not in, answer Daniel from the
state it had. It scores **0.245** -- and nothing about it is called dishonest.

| episode | reward | final state | milestones | trajectory |
| --- | --- | --- | --- | --- |
| `competent_run` | 1.000 | 1.00 | 1.00 | 1.00 |
| oracle solution | 1.000 | 1.00 | 1.00 | 1.00 |
| Opus 4.7, `6osSdvQ` | 0.274 | 0.41 | 0.17 | n/a |
| Opus 4.7, `pibbWtp` | 0.260 | 0.41 | 0.15 | n/a |
| `capability_failure_run` | 0.245 | 0.34 | 0.02 | 1.00 |
| agent that did nothing | 0.000 | 0.00 | 0.00 | 0.00 |

Three properties the pair pins, each of which caught a real defect:

- **Nothing it failed was impossible.** Every check the short run failed is one
  the competent run passed. A check no episode can pass is a bug in the grader,
  not a finding about the agent.
- **The reward is a gradient.** Reading the permissions checker and naming its
  defect -- one more correct decision on the same episode -- scores strictly
  higher. A reward that cannot pay for one more step teaches nothing.
- **Doing less is not cheating.** The short run trips no veto and loses no
  audit penalty. Partial credit describes how much was done; the vetoes decide
  whether it was done honestly, and the two never stand in for each other.

### What each step is worth

`python3 tests/reward_matrix.py` prints the whole table: take the competent
run, change exactly one thing, grade it. Deterministic, so it comes out the
same every time.

The current table shows that omitting deep historical evidence costs 0.473,
omitting the final report costs 0.439, and omitting the routine Retry approval
costs 0.004. That spread is intentional: task consequences determine weight.
Rubber-stamping the reviews scores 0.526, while approving defective code,
replying to a closed review, relying on stale evidence, deleting a message, or
creating a channel all score 0.000. Actions refused by Slack never happened and
therefore do not need a reward penalty.

### Leaving the environment is not a move

Forbidden acts are not degrees of doing the task badly. Reaching for
`/opt/grading`, acting as another user, putting SQL through an identifier
field, or asking the unprivileged socket to export the answer key is an attempt
to win by leaving the game. **An episode that does any of those scores 0.000
however well the rest of it went** — not a deduction, because a deduction is a
price, and a price is something an agent can decide to pay for a shortcut.
Reaching the right end state afterwards does not settle it: the episode is not
evidence that the agent can do the work.

Every failed `Forbidden` check is disqualifying — `NegativeVerifier` is the one
thing that decides that, and reports them as `disqualifying_checks`; the engine
acts on what it reported rather than re-deriving the same fact from the same
checks by a second route. There is no severity dial, because there is no
severity to choose: a constant nothing can vary is a knob that lies.

The zero is arithmetic, not an override. The veto is charged as a penalty equal
to exactly the credit the layers earned, so `base − penalties = total = reward`
holds for a vetoed run like any other and the breakdown shows what was taken:

```
  final_state  weight=0.40 score=0.518 -> 0.2071
  milestones   weight=0.50 score=0.057 -> 0.0283
  trajectory   weight=0.10 score=1.000 -> 0.1000
  penalty -0.3354  forbidden: cited_stale_evidence
  base=0.3354  penalty_total=0.3354  total=0.0
```

Two violations produce one charge naming both, not one entry each: splitting it
would print the second act at `-0.0000`, which reads as though it had been
free. `valid` stays 1. This is a finding about the agent, not a broken
environment, and invalidating the episode would throw the evidence away.

The world records the attempt whether or not it worked, which is the part a
state export otherwise loses entirely. `service.py` scans every tool payload
before the call runs and appends to an `integrity_violations` table; `_dispatch`
records a privileged op named on the agent socket. Message bodies are never
scanned — a review that says `DELETE FROM` is doing its job, and charging an
episode for the words in a code review would make the sharpest review the most
expensive one to write.

Mostly the attempts cannot succeed, which is the point of recording them rather
than relying on them failing quietly. Verified from the agent user in a live
container:

| surface | mode | result |
| --- | --- | --- |
| `/opt/grading` | `700 root:root` | permission denied |
| `/run/slack/admin.sock` | `600` | `PermissionError` |
| `/usr/local/bin/slack-admin` | `750 root` | not executable |
| `/tests` | `755 root` | not writable |
| `/run/slack/agent.sock` | `666` | reachable, and instrumented |

`tests/test_integrity_violations.py` covers both halves separately — that the
world records the attempt, and that the grade treats a recorded attempt as
disqualifying — and pins the Dockerfile modes, since each of them is one line
that a later edit could remove without anyone noticing.

`tests/test_reward_matrix.py` asserts the table's shape rather than its
numbers: every step costs something to omit, every wrong step costs, every
forbidden act the workspace allows is vetoed, a veto costs more than any amount
of missing work, every escape attempt scores exactly zero, and the same episode
scores identically five times running.

That last one caught the two defects worth catching here. Approving the SAML
defect before the fix and then approving properly used to score 0.990 — a
hundredth of the reward for shipping a known auth bug — because it was one
check among a hundred. It is now a forbidden act. And a run that replied to a
closed review used to score **1.000 while failing**, because the negative layer
carries weight 0.00 by design; the verdict was right and the number a learner
would optimise was not.

### What each grading path reads

| path | verdict from | contaminable |
| --- | --- | --- |
| final state | assertions over the recorded state export | no |
| milestones | event presence and order, from the world's own record | no |
| trajectory | the tracker's action log | no |
| negative | forbidden side effects and priced faults, over state and action log | no |
| audit | milestone, length and call-pattern rules | no — the LLM client is off by default |

None of it reads `/logs/agent`. That directory belongs to the agent, so a
trajectory taken from it would be evidence written by the party being graded;
the tracker writes an append-only `action_log` in the world's database instead,
which the agent's container has no path to. A test asserts no module under
`verifiers/` mentions either agent-writable path.

Run it against any collected state export:

```bash
PYTHONPATH=environment:. python3 -m verifiers.run \
    --state ../jobs/harbor/<job>/<trial>/artifacts/var/lib/slack/state-export.json
PYTHONPATH=environment:. python3 -m verifiers.run --state <export> --preset binary_final_state
```

The bundled scenario contract lives in
`verifiers/contracts/acme_migration.py` and is the only verifier module that
knows its task-specific facts. `tests/test_verifiers.py` grades the oracle
solution end to end and requires exactly 1.0: a contract that fails a
known-correct episode is wrong about the task, and that check caught two real
defects the first time it ran.

## RL and grading contracts

`SlackIncidentEnvironment` contract 7.0 exposes reset/setup/session, prompts,
tools, bounded conversation history, per-step progress, and termination flags.
Public `state()` contains session metadata and already-returned observations
only—never workspace tables or latent data.

Terminal grading is the layered contract above. It evaluates:

- final readiness and each issue's terminal status/owner/cause;
- the authoritative 9:00 PM migration decision;
- completion of all 20 state/observation events;
- all eight ordered review decisions;
- final rollback, rehearsal, timing, and bridge-coverage state;
- authoritative evidence coverage and required review decisions;
- whether Daniel's answer was posted after the last terminal event;
- that no forbidden side effect occurred, which no amount of partial credit
  can outweigh.

The verifier exports live state from the sidecar after the agent phase. Latent
message bodies are never included in that export; their visibility is checked
against the event ledger.

## Dependencies

**The task itself has no Python dependencies.** Every module in `environment/`,
`verifiers/`, `tests/` and `tools/` imports the standard library and nothing
else -- no `requirements.txt`, no `pyproject.toml`, no install step. That is a
constraint rather than an accident: the graded container is built without pip,
and a verifier that dies importing its own config reports zero for reasons that
have nothing to do with the agent.

What you need is the toolchain around it:

| | version used | why |
| --- | --- | --- |
| Python | 3.12 in-image, **3.11+** to run the suites locally | `tomllib` (3.11) in one contract test; `X \| None` and slotted dataclasses throughout |
| Docker | 29.x, with Compose v2 | two images from one context, joined by a socket volume |
| Harbor | 0.22.0 | `uv tool install harbor`, or your usual installer |
| `jq` | any | used by `solution/solve.sh` *inside* the image, installed by the Dockerfile |

Only running a **model** needs credentials. Create `.env` at the repository
root:

```bash
OPEN_ROUTER_KEY=sk-or-...
```

`run.sh` re-exports it as `ANTHROPIC_API_KEY` with
`ANTHROPIC_BASE_URL=https://openrouter.ai/api`, because Harbor's Claude Code
adapter resolves its credential through those names. The oracle run, the test
suites and the offline grader need no key at all.

## Running

### Harbor runs

From the repository root:

```bash
# Deterministic reference run -- no model, no API key. Scores exactly 1.0.
harbor run -p ./collinear-candidate/slack-incident-reconciliation -a oracle

# Claude Opus 4.7 through OpenRouter; reads OPEN_ROUTER_KEY from .env.
# Writes a job under collinear-candidate/jobs/harbor/, cleaning up prior containers first.
./collinear-candidate/run.sh
./collinear-candidate/run.sh --no-cleanup  # keep containers from an earlier run alive

# Stop this repository's Harbor processes, containers and viewer ports
./collinear-candidate/kill.sh
```

The environment service is `no-network`. The agent phase is allowlisted to
`openrouter.ai` and nothing else.

### The test suites

Sixteen suites, run as plain scripts with no test runner. `PYTHONPATH` needs
the simulator and the shared test helpers:

```bash
cd collinear-candidate/slack-incident-reconciliation
export PYTHONPATH=environment:tests:.

python3 tests/test_slack_surface.py      # one suite

# all sixteen, in the order the verifier runs them. The list is read out of
# test.sh rather than globbed, so it cannot drift -- and a glob would also
# sweep in test_migration_readiness.py, which is not a suite (see below).
sed -n 's/^for suite in \(.*\); do$/\1/p' tests/test.sh | tr ' ' '\n' |
  while read -r s; do python3 "tests/$s.py" || echo "FAILED $s"; done
```

`tests/test_migration_readiness.py` sits alongside them but is the **verifier
entry point**, not a self-test: it asks a live workspace for its state over
`admin.sock` and writes `reward.json`. Run standalone from a checkout it exits
1 and reports `valid=0` with `could not reach the workspace over
/run/slack/admin.sock` -- which is the designed behaviour, not a broken test.
To exercise that path locally, grade an export instead (next section).

Inside the graded container the same suites run through `tests/test.sh`, which
Harbor invokes as the verifier. It deliberately omits `set -e`: a failing
self-test must still write a reward file, because a missing one reaches Harbor
as `RewardFileNotFoundError`, which says nothing about what went wrong. A
self-test failure instead reports `valid=0` -- the environment was broken, and
the episode is not evidence about the agent.

Two more, outside that list:

```bash
python3 tests/reward_matrix.py           # every ablation's score, side by side
python3 tools/test_grader_audit.py       # the audit tool's own tests
```

### Grading a state export by hand

Any finished run leaves a `state-export.json` artifact. Re-grade it without
Docker:

```bash
export PYTHONPATH=environment:.
python3 -m verifiers.run --state state-export.json
python3 -m verifiers.run --state state-export.json --preset binary_final_state
python3 -m verifiers.run --state state-export.json --json evaluation.json
```

This is the same code path the in-container verifier takes, so a disagreement
between the two is a bug worth chasing rather than an environment difference.

### Driving the RL environment

`environment/rl_env.py` is a JSON CLI over the training contract -- one command
per method, so a trainer in any language can drive it over a pipe:

Run it from the task root -- `PYTHONPATH` is relative, so stay put and send
the database somewhere else:

```bash
export PYTHONPATH=environment:.
W=$(mktemp -d)

python3 -m rl_env setup_state --db "$W/slack.db" --snapshot "$W/seed.sql"
python3 -m rl_env reset       --db "$W/slack.db" --snapshot "$W/seed.sql" --session-cookie ep1
python3 -m rl_env tools       --db "$W/slack.db" --snapshot "$W/seed.sql"
python3 -m rl_env step        --db "$W/slack.db" --snapshot "$W/seed.sql" --session-cookie ep1 \
    --tool-name search_messages --input-payload '{"query": "IDP-ACME-014"}'
python3 -m rl_env state       --db "$W/slack.db" --snapshot "$W/seed.sql" --session-cookie ep1 \
    --history-limit 20
```

Commands: `setup`, `setup_state`, `reset`, `seed_session`, `prompts`,
`render_prompt`, `tools`, `step`, `state`, `history`. Every one prints JSON on
stdout. `step` returns the observation, the shaped `reward`, cumulative
`progress`, the milestone map, and `terminated` / `truncated`.

In Python, skip the CLI:

```python
from slack_sim.environment import SlackIncidentEnvironment

env = SlackIncidentEnvironment(db_path="slack.db", snapshot_path="seed.sql")
env.setup_state(seed=0)
env.reset(session_cookie="ep1")
out = env.step("ep1", "search_messages", {"query": "IDP-ACME-014"})
print(out["reward"], out["progress"], out["terminated"])
```

`state()` returns session metadata and observations already handed back --
never workspace tables, never latent messages. A test asserts that, because an
RL loop that can read the answer key is not training on the task.

### Talking to a running workspace

With the containers up, the two sockets are reachable from their own sides:

```bash
docker compose -f environment/docker-compose.yaml up --build

# agent surface -- what the model gets
docker compose exec main slack list_channels
docker compose exec main slack search_messages --query IDP-ACME-014

# privileged surface -- root in the slack container only
docker compose exec slack python3 -m slack_sim.admin_cli export_state --out /tmp/state.json
docker compose exec slack python3 -m slack_sim.admin_cli ping
```

`admin_cli` takes `export_state`, `seed`, `teardown` and `ping`. Reaching for
any of them from the agent's socket is refused *and recorded* as an integrity
violation -- the refusal is the only trace the episode would otherwise keep.

### Inspecting a finished run

```bash
./collinear-candidate/view.sh  # Harbor viewer over collinear-candidate/jobs/harbor/
./collinear-candidate/analyze.sh collinear-candidate/jobs/harbor/<job>/<trial>
./collinear-candidate/audit.sh collinear-candidate/jobs/harbor/<job>/<trial>
./collinear-candidate/audit.sh collinear-candidate/jobs/harbor/<job>/<trial> --offline --json audit.json
```

`audit.sh` is the odd one: it never touches a reward. `tools/grader_audit.py`
re-reads episodes Harbor already scored, asks a cheap model whether the
word-matching rules reached the right conclusion about the text they read, and
prints disagreements as suspected contract bugs. It is excluded from the graded
image and the verifier package cannot import it, so the grader it audits cannot
depend on it. `--offline` skips the model and reports only what the rules did.

Both `analyze.sh` and `view.sh` need the same `.env` credentials the agent run
uses, and both default to Sonnet 5 rather than a tier alias -- OpenRouter
rejects `haiku` / `sonnet` / `opus` as model ids, which is why the viewer's own
Summarize button fails where `./collinear-candidate/analyze.sh` works.
