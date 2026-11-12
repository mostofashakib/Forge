# Forge working plan: from task generator to self-improving factory

Forge's goal is to be an RL environment factory. It produces environments, simulated
users, and verified tasks, and turns them into datasets that move frontier models
forward, including in domains where data is hard and expensive to get.

Today Forge generates and verifies tasks well. What it lacks is the loop: nothing it
learns from running models changes what it generates next. This plan builds that loop
in small steps. Each milestone ends with a number you measure before you start the
next one.

LLMs write, review, and solve the tasks, so every step is a place for slop to get in.
The loop only counts as working when a trained model gets better on tasks the loop
never touched. Forge's own numbers going up is not enough. M4 is a set of experiments
to find the settings that produce that gain, and M5 chains them into a loop that runs
round after round.

The cloud deployment in `TASKS.md` stays on its own track. Nothing here depends on it.
Every milestone runs on a laptop with a local Ollama model.

---

## Where things stand

What exists and what this plan reuses:

| Piece | Where | State |
|---|---|---|
| Task factory: taxonomy, slot plan, writer, static check, golden pass^k, cross-family review, contamination check | `forge/taskfactory/` | Works. The strongest part of Forge |
| Fresh seeded sessions with `step()`, `evaluate()`, `fingerprint()` for state, CLI, browser, and container apps | `forge/taskfactory/runner.py`, `forge/taskfactory/runners/` | Works. Only golden solutions use it |
| Immutable, versioned batches | `TaskBatch`, `GeneratedTask` in `backend/app/models.py` | Works |
| Per-task step budget (twice the golden length) | `SyntheticTask.step_budget` | Stored, never enforced on a model |
| Seven failure modes | `forge/runtime/loss_analysis.py` | Defined. No production code calls it |
| Safe expression evaluator | `forge/runtime/safe_expression.py` | Works. Useful for domain checks in M5 |
| Export writers (SFT, DPO, GRPO, failures, trajectories) | `backend/app/services/export_writers/` | Work on agent-run episodes, not on factory tasks |
| Synthetic Data Engine | `backend/app/api/synthetic.py`, `/environments/[env]/synthetic` | Overlaps the factory and is unverified (see M1) |

The gaps, in the order this plan closes them:

1. Two ways to make data, with only one verified. (M1)
2. Difficulty means golden-solution length. Nobody measures how often a real model solves a task. (M2)
3. Datasets come from golden solutions and rejected drafts, not from real model attempts. (M3)
4. Nothing shows that Forge data makes a model better, or which settings make it better. Nothing guards against checks a wrong solution passes. (M4)
5. Results never change what the writer produces next, and no run chains generation, training, and evaluation round after round. (M5)
6. Every domain is cheap: SaaS mocks, shell, browser, public repos. Check types cover rows, values, shell exit codes, and DOM text. (M6)

---

## Decisions to make before starting

Answer these before M2 starts. Each one changes the code.

**D1. How the target model acts.**
- Option A (recommended): a small policy built on `forge/extraction/llm_client.py`. It already supports Ollama and every other configured provider, and it returns structured output. The model picks the next `GoldenStep` in the same action format the writer uses, so one policy covers state, CLI, and browser tasks.
- Option B: reuse the agents in `forge/runtime/agents/` (Anthropic, OpenAI, vLLM). They use native tool calling, which gives more realistic trajectories, but they speak the `ForgeEnv` interface, have no Ollama adapter, and don't fit CLI or browser sessions without new glue.
- Trade-off: A is faster to build and runs locally. B gives trajectories closer to how production agents call tools. Starting with A keeps M2 small. The `Policy` protocol in M2 lets you add B later without touching calibration code.

**D2. The first target model.** Pick one local model you would train (for example a Qwen or Gemma size you plan to fine-tune in Autotune). Calibration results only mean something for the model they were measured on.

**D3. Attempts per task.** Start with 8. With 8 attempts, solve rate moves in steps of 12.5%, which is enough to sort tasks into too easy, frontier, and unsolved. Raise it only if M3 data shows too many tasks on the band edges.

**D4. The frontier band.** Start with "solved at least once and failed at least once" (0 < solve rate < 1). Tighten it after M2 shows the real distribution. Keep the bounds as named constants, not settings, until data justifies a knob.

**D5. The M6 domain.** Recommended: accounting reconciliation (ledgers, invoices, bank feed). Data is expensive, and code verifies the result: balances match or they don't. Clinical and legal work wait until expert review is available.

### Answer D6, D7, and D8 before M4 starts

M4 does not start until all three of these have written answers in this file. Every M4 experiment depends on them.

- **D6:** which transfer environment and which external benchmark to use.
- **D7:** the compute budget. M4 needs about 28 training arms, and 3 seeds each would mean about 84 runs. The plan suggests one seed to screen each experiment and three seeds only for the comparison that decides it.
- **D8:** how Forge hands data to Autotune. It stays manual through M4.

Details for each:

**D6. How to measure capability outside Forge.** Pick one transfer environment the loop never trains on, and one public benchmark close to the environments you train on (for example a terminal benchmark for CLI work, or a tool-use benchmark for app work). Autotune runs the benchmark. A gain on Forge's held-out set that doesn't show up on either of these is a warning sign.

**D7. Compute budget.** M4 has about 28 arms that each need a training run. Three seeds per arm means about 84 runs. Recommended: screen each experiment with one seed per arm, then rerun only the deciding comparison with three seeds. Set a ceiling in GPU hours before starting, and log what each experiment spends.

**D8. The Forge to Autotune handoff.** Forge exports a dataset with a manifest. Autotune trains and returns a model id. The trained model has to be reachable by the M2 policy (through Ollama or vLLM). Do this by hand through M4. Automate it in M5 only after one manual round works.

---

## Rules for the work

### One branch per milestone, merged only when nothing broke

Every milestone follows the same cycle:

1. **Branch** from the latest `main`: `factory/m1-merge-synthetic`, `factory/m2-calibration`, `factory/m3-datasets`, `factory/m4-experiments`, `factory/m5-loop`, `factory/m6-domain`.
2. **Build** the milestone on that branch, committing as you go.
3. **Check that core features still work.** All of these must pass on the branch:
   - `uv run pytest -q` (the full suite, including `tests/e2e/test_critical_flows.py`)
   - `npm --prefix frontend run build`
   - `npm --prefix frontend test`
   - `npm --prefix frontend run lint` reports no errors in changed files
   - the milestone's own "Done when" checks
   - a manual pass through the core flows in the running app: create or start an environment, run a task factory batch, start an agent run, and export a dataset
4. **Compare with `main`.** A test that fails on the branch must also fail on `main` for the same reason, or the branch broke it. Record the suite counts for both in the log.
5. **Merge** into `main` only when every check passes. Log the merge commit.
6. **Move on** to the next milestone, branching from the new `main`.

A milestone that breaks a core feature does not merge until the break is fixed.

### How to build

- TDD on every change. Write the failing test first, then the code. Keep `uv run pytest -q` and `npm --prefix frontend run build` green.
- No fake data in production paths. Fakes live in tests only. A missing model or provider fails with a clear error that names the setting.
- Batches stay immutable. New results go in new tables keyed to a batch, never into `task_batches` or `generated_tasks`.
- Each milestone's "Done when" is checked by running it, and the evidence (command plus last lines of output, or the measured numbers) goes in the milestone's log section at the bottom of this file.

---

## M1. Fold the Synthetic Data Engine into the factory

**Goal.** One way to make data, and it's verified. Keep the two ideas from the Synthetic Data Engine that the factory lacks and delete the rest.

**Why.** The engine writes shell commands blind, without seeing their output. No task-specific check grades the result. A saved manifest silently replaces the chosen model in every CLI run on that environment (`backend/app/worker/container_episode.py:78`). Its difficulty scale (level 5 = 15 to 20 commands) contradicts the factory's (levels 4 and 5 = 30+ steps).

**What to keep.**
- Edge-case types: `boundary_conditions`, `permission_errors`, `missing_deps`, `conflicting_state`, `recovery`.
- The intent behind `failure_cases`: failed trajectories for preference and failure datasets. M2 and M3 deliver these from real, graded model attempts.

### M1.1 Edge cases become a slot attribute

Work:
1. `forge/taskfactory/schemas.py`: add `EdgeCase` (a Literal of the five types) and `FAMILY_EDGE_CASES`, mapping each family to the types that make sense there. `state`: `boundary_conditions`, `conflicting_state`, `recovery`. `cli`: all five. `browser`: `boundary_conditions`, `conflicting_state`, `recovery`.
2. `TaskDraft` and `SyntheticTask` get `edge_case: EdgeCase | None` and `edge_case_seed: list[str]`. The second field names the seed elements that plant the condition: record ids for state, setup command indexes for CLI, page paths for browser.
3. `forge/taskfactory/slots.py`: `Slot` gets `edge_case`. `plan_slots(taxonomy, count, edge_cases=())` cycles slots through the requested types plus "none". With two types requested, about a third of the slots get each. The plan stays exact and deterministic.
4. `forge/taskfactory/static_check.py`: a slot with an edge case is rejected when the draft's `edge_case` doesn't match, when `edge_case_seed` is empty, or when it names a seed element that doesn't exist. Code checks this, not the LLM.
5. `forge/taskfactory/writer.py`: the prompt tells the writer to plant the edge case in the seed and to make the golden solution handle it. Move the five descriptions from `synthetic.py` into the writer prompt.
6. `forge/taskfactory/review.py`: for edge-case slots, the reviewer also judges whether the seed truly sets up the condition and whether the golden solution handles it. Add the field to `ReviewVerdict`.
7. `POST /api/task-factory/batches` takes `edge_cases: list[str]`, validated against the environment's family. The generator page gets the five checkboxes, filtered by family.

Tests first:
- `test_slots.py`: edge cases spread evenly across slots, slots without a request get none, and an unknown type raises.
- `test_static_check.py`: rejects a missing edge case, an empty `edge_case_seed`, and a seed reference that doesn't exist. Accepts a correct draft for each family.
- `test_writer_and_review.py`: the prompt includes the edge-case description only for edge-case slots, and a reviewer verdict that rejects the edge case rejects the task.
- `test_pass_k.py`: a CLI draft with a `conflicting_state` seed (a file already exists) passes when the golden solution handles it and fails when it doesn't.
- API test: a `permission_errors` request for a state environment returns 422.

### M1.2 Delete the Synthetic Data Engine

Work:
1. Delete `backend/app/api/synthetic.py` and its router registration.
2. Delete the replay branch in `backend/app/worker/container_episode.py` (`_cli_agent` builds the model agent only) and `ReplayCliAgent` in `forge/envgen/agents/cli_agent.py`.
3. Delete `frontend/app/environments/[env_name]/synthetic/`, `SyntheticEpochCard.tsx`, and their links in `NewRunModal.tsx` and the environment page.
4. Remove the "Synthetic Data Engine" section from `README.md` and add edge cases to the Task Factory section.
5. Leftover `synthetic_replay.json` files on disk are ignored from now on. Delete them by hand. No migration code is needed.

Tests first:
- A worker test: a CLI run on an environment with a `synthetic_replay.json` file present uses the requested model agent. It fails today, which proves the takeover bug.
- Delete the tests that cover only the removed endpoints.

**Done when.**
- A 30-task CLI batch with `conflicting_state` and `recovery` requested finishes, and every edge-case task passes static check, pass^k, and review. Log the batch version and the counts per edge case.
- `grep -rn "synthetic_replay\|ReplayCliAgent" backend forge frontend/app` returns nothing.
- Full test suite and frontend build pass.

---

## M2. Calibrate tasks against a target model

**Goal.** For every task in a batch, measure how often a chosen model solves it, using the same seeds and checks that pass^k uses.

**Why.** Frontier data comes from tasks a model solves sometimes. Golden-solution length doesn't tell you which tasks those are. This is the measurement every later milestone depends on.

### M2.1 The policy and the attempt loop

Work:
1. `forge/taskfactory/policy.py`: a `Policy` protocol with `next_step(objective, tools, history) -> GoldenStep | Submit`, and an `LLMPolicy` built per D1. The history is the list of past steps with their `StepResult.output`.
2. `forge/taskfactory/attempt.py`: `run_attempt(runner, task, policy) -> Attempt`. It opens `runner.session(task)`, records the pre-state check results, calls `policy.next_step` until the policy submits, hits `task.step_budget`, or a step errors, then runs `session.evaluate(task.checks)`.
   - `Attempt` holds: the steps with outputs, each check's outcome, `passed` (every check passes), and the termination reason (`submitted`, `budget`, `step_error`, `policy_error`).
3. `runner.session()` takes a `TaskDraft` today. Widen the protocol to take the shared fields (seed, checks) so a stored `SyntheticTask` runs without being turned back into a draft.
4. A failure label per attempt, in code: which checks failed, the termination reason, and whether the model touched a tool the golden solution never used. Check whether `LossAnalyzer` fits these inputs. If it does, call it. If its inputs don't match, record the simple labels now and note the gap in the log. Don't force-fit it.

Tests first (the two anchors matter most):
- A policy that replays the golden solution passes every attempt, for state, CLI, and browser runners.
- A policy that submits at once fails every attempt. pass^k guarantees at least one outcome check fails before the golden solution, so this must hold.
- A policy that never submits stops at `step_budget` with termination `budget`.
- A policy that raises ends the attempt with `policy_error`. It must never count as a pass.
- `LLMPolicy` with the fake LLM from `tests/fake_llm.py` turns structured output into `GoldenStep`s and rejects a tool the environment lacks.

### M2.2 Storage, job, and API

Work:
1. New tables in `backend/app/models.py`:
   - `calibration_runs`: id, batch_id, target_model, attempts_per_task, status, error, created_at, completed_at.
   - `calibration_attempts`: run_id, task_id, attempt_index, passed, termination, steps_json, checks_json, failure_labels_json.
   - `create_all` adds them. No column migration is needed.
2. `backend/app/worker/calibration.py`: a Celery job that runs each task's attempts under the environment lock, in chunks like the factory's `PROCESS_CHUNK`, and reports progress over the existing progress channel.
3. API under `/api/task-factory/batches/{batch_id}/calibrations`: `POST` (target model, attempts), `GET` list, `GET /{id}` with per-task solve rates and a summary (counts of too easy, frontier, unsolved, by category and by difficulty).
4. The job refuses to start if the target model's provider isn't configured, and names the missing setting.
5. `frontend/app/generator/[batchId]/page.tsx`: a calibration panel. Start a run, watch progress, see the solve-rate histogram and each task's rate. Open a task to see its attempts.

Tests first:
- The job writes one attempt row per task per attempt, and a crash midway leaves the run `failed` with its error, not `complete`.
- The summary groups solve rates correctly by band, category, and difficulty.
- `POST` with an unconfigured provider returns a clear error and starts nothing.
- Two calibrations of the same batch with different models are stored separately, and the batch rows are unchanged.

**Done when.**
- One real batch (at least 50 tasks, one environment) is calibrated against the D2 model with 8 attempts each.
- The log records the solve-rate histogram, the band counts, and the share of tasks per difficulty level in each band.
- A spot check of 5 passing and 5 failing attempts by a person agrees with the grade in every case.

**Metric to record.** How well golden-length difficulty predicts solve rate. If level 1 and level 4 tasks have similar solve rates, length is a weak signal, and that is the evidence M4 needs.

---

## M3. Datasets from real attempts

**Goal.** Export training data built from graded model attempts, filtered to the frontier band.

**Why.** Today's factory exports are golden solutions (SFT) and rejected task drafts (pairs of writer outputs). Neither shows a model succeeding and failing at the same task. Calibration produces exactly that.

Work:
1. `forge/taskfactory/bands.py`: `FRONTIER_MIN` and `FRONTIER_MAX` constants (D4) and `classify(solve_rate) -> too_easy | frontier | unsolved`.
2. Export formats from a calibration run, added to the batch export route as `?calibration=<id>&format=<name>`. Match the shapes of the writers in `backend/app/services/export_writers/`:
   - `grpo`: one group per frontier task, with the prompt, the n attempts, and their 0 or 1 rewards.
   - `preference`: for each frontier task, pairs of one passing and one failing attempt. The failing side carries its failed checks as the reason.
   - `sft`: passing attempts from frontier tasks. Add golden solutions for unsolved tasks only if you choose to, and label them as golden.
   - `failures`: failing attempts with failed checks and failure labels.
3. The dataset manifest records the calibration run id, target model, attempts per task, band bounds, and `source: model_rollouts`. Every download passes the existing safety gate.
4. An "unsolved" list on the batch page. A 0% task is either hard or unfair, because pass^k only proves it's solvable. A person marks each one `hard` or `unfair`. Unfair tasks are excluded from later exports. Nothing gets deleted automatically.

Tests first:
- `classify` at each band edge.
- `grpo` includes frontier tasks only, and each group has exactly n attempts.
- `preference` makes no pair for a task with only passes or only fails. It never invents a side.
- The manifest names the calibration run and model.
- A task marked `unfair` is missing from every export.

**Done when.**
- The M2 calibration exports in all four formats.
- One format loads in Autotune (GRPO or DPO) without changes on the Autotune side. Log the command and the result.

---

## M4. Experiments: find what makes the model better

**Goal.** Find the loop settings that produce data a model gets better from, and the settings that produce slop. Each experiment asks one question, changes one thing, and ends with a decision.

**Why.** Frontier yield, review pass rates, and solve rates are Forge's own numbers. A loop that pushes them up and leaves the model no better somewhere else is producing slop. Typical ways it goes wrong:
- tasks that look hard because they are ambiguous
- checks that a wrong solution passes
- data that teaches Forge's quirks instead of a skill

### M4.0 Instruments, built before any experiment

The experiments are only as trustworthy as these.

1. **Held-out set.** A frozen set of at least 100 tasks per environment. A different writer model than the loop uses writes them, and a person audits every one. They are never used for training, steering, or band tuning. To evaluate a model, you calibrate it on this set with the M2 code. Evaluation and data generation then use the same grader.
2. **Transfer set (D6).** Held-out tasks from an environment the loop never trains on. A gain here means the model learned a skill, not Forge's habits.
3. **External benchmark (D6).** One public benchmark, run in Autotune. A drop here is a regression, even if Forge's numbers rise.
4. **Mutation test.** For every accepted task, run two broken versions of the golden solution: one with its last state-changing step removed, and one with one argument changed. At least one check must fail on each. If a broken solution passes, the task is rejected at a new `mutation` stage after pass^k. This is the main code-level guard against weak checks.
5. **Slop metrics per batch:** review rejection rate, tasks marked `unfair`, near-duplicate rate (overlap in titles and objectives), mutation-stage rejections, and the share of passing attempts a person grades as not done. Shown on the batch page and recorded in the manifest.
6. **Human audit.** Each round, a person grades 20 random accepted tasks and 20 random passing attempts. If grades agree with the checks less than 90% of the time, the round stops.
7. **Noise floor.** Train on the same data with 3 seeds and evaluate each. The spread across seeds sets the smallest gain worth believing. A gain counts only when it is bigger than twice that spread.

Tests first:
- The mutation stage rejects a task whose checks pass with a dropped step or a changed argument, and accepts one where a check fails on both.
- A task with a single golden step still gets a mutation run (the dropped-step version is the empty solution, which pass^k already proves fails).
- Held-out batches are refused by every export and left out of every steering statistic.
- Near-duplicate rate is 0 for distinct tasks and flags a task that repeats another with renamed values.

### M4.1 Code the experiments need

1. **Steering.** Every batch builds a fresh taxonomy today. Steering needs categories to carry over between rounds. So `POST /api/task-factory/batches` takes an optional `parent_calibration_id`. With it, the batch copies the parent's taxonomy into its own row and skips the taxonomy step.
2. `forge/taskfactory/slots.py`: a pure function `plan_steered_slots(taxonomy, stats, count, edge_cases)`. From each category's and level's band counts:
   - categories and levels with more frontier tasks get more slots
   - a mostly too-easy category moves up one difficulty level
   - a mostly unsolved category moves down one level
   - every category keeps at least one slot
3. **Writer feedback** (for E9): an optional line per category in the writer prompt with its solve rate and most common failure label.
4. The batch row records `parent_calibration_id` and the steering variant, so every result traces back to what made it.

Tests first:
- `plan_steered_slots` gives more slots to frontier-heavy categories, shifts too-easy and unsolved categories by one level, keeps every category, and returns exactly `count` slots.
- With equal stats it matches `plan_slots`.
- A batch with a parent calibration copies the taxonomy and makes no taxonomy LLM call.
- The writer prompt includes feedback lines only when the variant asks for them.

### M4.2 The experiments

Run them in this order, since early results set what later ones hold fixed. Unless an experiment says otherwise, these stay the same: the target model (D2), the Autotune training recipe, the held-out and transfer sets, attempts per task, and the budget per arm (D7).

Every capability measurement reports four numbers: held-out solve rate, transfer solve rate, external benchmark score, and human audit agreement.

**E1. Does Forge data help at all?**
- Arms: no training (baseline), SFT on golden solutions, GRPO on frontier tasks from one calibrated batch.
- Decision: if neither training arm beats the baseline beyond the noise floor, stop. Fix data quality before anything else. Steering a loop whose data doesn't help only makes more of it.

**E2. Which band trains best?**
- Arms, all the same size: every task, 0 < solve rate < 1, a narrow band (0.125 to 0.5), and the hard edge only (0.125 to 0.25).
- Decision: sets `FRONTIER_MIN` and `FRONTIER_MAX` (D4) from results, not guesses.

**E3. Which data format gives the most gain per task?**
- Arms, on the same tasks: SFT on passing attempts, DPO on pass and fail pairs, GRPO groups, SFT followed by GRPO.
- Decision: the default export for M5.

**E4. How much data before gains level off?**
- Arms: 100, 300, and 1,000 frontier tasks (scaled down if D7 requires it).
- Measure: gain per task and cost per point of gain.
- Decision: round size for M5.

**E5. Breadth or depth?**
- Arms, the same size: many categories with few tasks each, few categories with many tasks each, and the better of those two with and without edge cases.
- Watch: depth often lifts held-out scores but not transfer. Transfer decides.
- Decision: slot plan defaults and the edge-case share.

**E6. Does each validation stage earn its cost?**
- Arms, with the same writer and slots: full validation, no LLM review, no mutation test.
- Measure: slop metrics, and capability after training on each arm's data.
- Decision: keep or drop each stage based on capability, not on how many tasks it rejects.

**E7. How strong does the writer need to be?**
- Arms: a writer stronger than the target, one the same size as the target, and the target model itself.
- Measure: slop metrics, frontier yield, and capability gain.
- Why: if the target writes tasks good enough to train itself, the loop bootstraps without a stronger model. This is the RSI question in its smallest form. A negative result tells you the loop needs a stronger writer for now.

**E8. Does steering by slot reweighting help?**
- Arms: steered and unsteered batches from the same parent calibration and taxonomy, 3 paired runs.
- Measure: frontier yield, cost per frontier task, and capability gain after training on each.
- Watch: yield rising without capability gain means steering finds tasks that are hard for the wrong reasons.

**E9. Does writer feedback add to steering?**
- Arms: E8's winner, with and without the per-category feedback line.
- Same measures as E8.

**E10. Does training teach the model to exploit checks?**
- A person grades 50 passing attempts on the held-out set from the best model of E1 to E9, and 50 from the baseline.
- Measure: the share that checks pass but the person grades as not done, before and after training.
- Decision: a rise means the checks have holes the model learned to use. Fix the checks, and add the pattern to the mutation test, before M5.

**Done when.** Every experiment has a log entry with its arms, ids (batches, calibrations, exports, Autotune runs, models), numbers, and decision. A "Settings for M5" table in the log records the chosen band, data format, round size, slot plan, edge-case share, validation stages, writer, and steering variant. Each setting points to the experiment that chose it. Negative results count as done.

---

## M5. Chain the loop

**Goal.** Run the whole loop for several rounds, with each round's trained model as the next round's target, and show that capability rises round over round.

**Why.** M4 finds good settings one piece at a time. Chaining shows whether the gains stack, where they level off, and what breaks first when the loop runs on its own output.

**One round** (round r, starting from model M(r-1), with M(0) the base model):
1. **Generate** a batch with the M4 settings, steered from round r-1's calibration. Round 1 is unsteered.
2. **Validate** with the stages E6 kept, including the mutation test.
3. **Calibrate** the batch against M(r-1).
4. **Export** frontier data in the format E3 chose.
5. **Train** M(r) in Autotune.
6. **Evaluate** M(r) on held-out, transfer, and the external benchmark. Run the human audit and record the slop metrics.
7. **Gate.** The next round starts only when all of these hold:
   - the held-out gain over M(r-1) beats the noise floor
   - transfer and the external benchmark did not drop
   - human audit agreement is at least 90%
   - slop metrics stay within the bounds set in M4

   Otherwise the loop stops and logs which condition failed.

**Two choices the first rounds settle.**
- **How to train M(r).** Continue from M(r-1) on the new data, or retrain from the base model on every round's data so far. Try both for rounds 1 and 2 and keep the better one. Continuing is cheaper but risks forgetting.
- **When the held-out set runs out.** Once a model solves more than 80% of it, it stops measuring progress. Before that point, a person audits a fresh, harder held-out tier written by the non-loop writer.

### M5.1 Two rounds by hand

Run rounds 1 and 2 with the existing API, CLI, and Autotune commands, logging every id. Write down each manual step and where it was slow or easy to get wrong. That list decides what M5.2 automates.

### M5.2 Automate the round

Work:
1. Tables `loop_runs` and `loop_rounds`. A round records its batch, calibration, export, Autotune run id, model id, evaluation results, slop metrics, audit result, gate result, and the reason it stopped.
2. `forge/loop/gate.py`: the gate as a pure function from a round's results to `continue` or `stop` with reasons.
3. A Celery chain that runs one round's steps in order. Training goes through Autotune's interface (D8). The human audit step pauses the round until a person submits grades on the batch page or through the CLI.
4. `forge loop start`, `forge loop status`, and `forge loop resume` in `forge/cli/`.

Tests first:
- The gate stops on each failing condition alone, names that condition, and continues only when all pass.
- A round never starts before the previous round's gate says `continue`.
- A failed Autotune run, or an evaluation that fails to run, marks the round `failed` and stops the loop. It never counts as a pass.
- An integration test with fakes for the LLM and Autotune runs two rounds end to end and proves no held-out task reaches any export.

### M5.3 Run it

Run 4 to 5 rounds with the automated loop.

**Done when.** The log holds a per-round table and chart of held-out, transfer, external benchmark, slop metrics, and cost. It ends with a verdict: does capability rise, after how many rounds it levels off, and what stopped the loop.

**After M5, if E7 was positive.** Run a chain where M(r) also becomes the writer for round r+1. That is the loop improving the model that improves the loop. Run it only when the guards above held through a full M5 chain.

---

## M6. First hard domain: accounting reconciliation

**Goal.** One domain where data is expensive, verified by checks that encode domain rules, run through the M5 loop.

**Why.** The vision includes domains where producing data is hard. In those, the verifier is the bottleneck. Everything upstream is only as good as the checks.

Work:
1. **Environment.** Generate a custom environment with the existing generator: general ledger, chart of accounts, invoices, payments, bank feed, and period close. It must pass the determinism gates like any custom environment.
2. **Domain rule pack.** A versioned file of rules written or approved by someone who knows the domain (for example "every posted journal entry balances", "a bank line matches at most one payment", "no posting into a closed period"). The writer must cite the rules a task depends on, and the reviewer checks the citations.
3. **New check kinds** in `schemas.py` and `state_checks.py`, built on `forge/runtime/safe_expression.py`:
   - `computed`: an expression over state compared to a value within a tolerance (for example the sum of an account's lines equals its balance to the cent).
   - `all_records`: every record matching a filter satisfies an expression (for example every entry in period 2023-10 has debits equal to credits).
   - Both run in code, so pass^k and calibration grade them the same way as today's checks.
4. **Human agreement.** A person grades 30 tasks and 30 attempts by hand. Record how often the checks agree with the person. That rate is the verifier quality number for this domain.

Tests first:
- `computed` passes within tolerance, fails outside it, and fails clearly on a missing path or bad expression (never passes by default).
- `all_records` fails when one matching record breaks the rule, and passes when no record matches only if the check says an empty match is allowed.
- Static check rejects a draft that cites a rule missing from the pack.
- False positives: a ledger that balances in total but has one unbalanced entry fails `all_records`.

**Done when.**
- A batch of at least 50 reconciliation tasks is accepted, passes the mutation test, and is calibrated and exported.
- Human agreement is at least 90%. If it's lower, the log lists the disagreements and the rule or check behind each one, and fixing those comes before scaling up.
- An M5 chain of at least 2 rounds runs in this domain and passes its gates. It needs its own held-out set and a transfer check (another domain with rules, or a public finance benchmark).

---

## Not now, and when to start

| Item | Why not now | Start when |
|---|---|---|
| Environment generator learns from build and review failures | The 10-environment cap means too few builds to learn from | After M5. First check what build failures the pipeline already stores. Add a table only if nothing queryable exists today |
| Persona realism checks | Measuring realism needs real user transcripts to compare against, and there are none | A domain needs personas to drive tasks, and you have a sample of real conversations for it |
| Native tool-calling policy (D1 option B) | Option A covers calibration with less code | E3 shows trajectory format matters to training results in Autotune |
| Learned failure classifier | Simple labels from M2 come first | E9 shows failure labels help the writer, and the simple labels are too coarse |

---

## Milestone order and what each one proves

| Milestone | Proves | Depends on |
|---|---|---|
| M1 | Forge has one verified way to make data, with edge cases | Nothing |
| M2 | Forge measures how hard each task is for a real model | M1 (for the clean CLI path), D1 to D3 |
| M3 | Forge turns those measurements into frontier datasets | M2 |
| M4 | Forge data makes a model better outside Forge, and which settings do it without slop | M2, M3, D6 to D8 |
| M5 | The gains stack when the loop runs round after round on its own output | M4 |
| M6 | The loop works in a domain where data is expensive | M5, D5 |

---

## Log

Record evidence here as each milestone finishes: the step, the command, and the last
lines of output, or the measured numbers.

### M1

Branch `factory/m1-merge-synthetic`. Commits: `3a2b99f` (edge cases), `78e7adc` (Synthetic Data Engine removed).

Baseline on `main` before M1: `uv run pytest -q` 2779 passed, 2 skipped. Frontend tests 11 passed. Lint 0 errors, 6 warnings.

On the branch:
- `uv run pytest -q`: 2819 passed, 2 skipped. The 40 new tests cover slot planning, the static check, writer and reviewer prompts, the pipeline, the API, the job, the migration, and the replay takeover.
- `npm --prefix frontend test`: 16 passed, 0 failed.
- `npm --prefix frontend run build`: compiled. One earlier run failed while downloading Google fonts (network), and the rerun passed.
- `npx eslint` on every changed frontend file: exit 0.
- `grep -rn "synthetic_replay\|ReplayCliAgent" backend forge frontend/app`: no matches.
- The replay takeover was proven before the fix: `test_a_leftover_synthetic_replay_file_never_replaces_the_requested_agent` failed with `ReplayCliAgent` in place of the requested agent.

Change from the plan: `permission_errors` is not offered. CLI tasks run as root, and the other families have no permission model the factory can rely on. `missing_deps` is not offered for browser environments.

Core flows against the running stack (API on port 8100, since vidQ holds 8000 and 3000):
- `/api/health`: database, Redis, Docker, worker, and LLM all ok.
- Create a CLI environment (`m1-shell`): queued, then running.
- Agent run (random agent, 1 episode): completed, 1 episode, no error.
- Export (`trajectories`, `sft_pairs`): completed, and the manifest was written. Download returned 200. It had 0 rows, because export reads the rollout `episodes` table and agent runs write `agent_episodes`. That behavior predates M1, and M1 does not touch export.
- Task factory batch `tb_a9b84a86232a` (30 tasks, `conflicting_state` and `recovery`, k=3): **failed on the first writer call**. Anthropic returned "credit balance is too low". The job recorded the error and saved nothing.

Open: the 30-task "Done when" batch, which needs a working writer model from a family other than the validator's. M1 does not merge until it passes.

### M2

### M3

### M4

#### Settings for M5

| Setting | Value | Chosen by |
|---|---|---|
| Frontier band | | E2 |
| Data format | | E3 |
| Round size | | E4 |
| Slot plan and edge-case share | | E5 |
| Validation stages | | E6 |
| Writer | | E7 |
| Steering variant | | E8, E9 |

### M5

### M6
