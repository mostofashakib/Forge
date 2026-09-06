# Forge Example RL Tasks

Two deterministic, self-contained Harbor tasks. Each tests whether an agent can
inspect state it was not told about, act through tools, respect the boundaries
of what it was asked to do, and report honestly — and each grades that with a
layered verifier rather than a pass/fail check.

| Task | Difficulty | Core challenge |
| --- | --- | --- |
| [`slack/`](slack/) | hard | Drive a live enterprise cutover through staged reviews where approving one revision releases the next, evidence arrives only after the action that provokes it, and a late rehearsal invalidates readiness the agent has already established |
| [`task_manager/`](task_manager/) | medium | Hand over a departing teammate's queue: find the whole set, branch each record on its own fields, append a label without replacing the ones already there, and touch nothing else |

Each directory is a complete Harbor task — `task.toml`, environment, solution,
tests and verifiers — and stands alone. There is no shared package between them
and no install step; both import the standard library and nothing else.

## How they are built

The same architecture in both:

- **Two containers from one image build.** The agent's container holds the
  client half only: no seed, no service, no database. The world runs beside it
  and they share nothing but a directory holding two Unix sockets — one for the
  agent, one privileged for lifecycle and verifier export.
- **Tools arrive as an MCP server** declared in `task.toml`, so the agent is
  handed the workspace rather than told to go looking for it.
- **A virtual clock in SQLite**, so time is a property of the workspace and two
  runs of the same episode are byte-identical.
- **A scenario engine driven by data** — events, triggers and latent rows — so
  the world reacts to what the agent does instead of replaying a script.
- **A weighted, layered verifier** with presets and a veto layer, reading only
  world-side evidence: the tracker's own append-only action log, never the
  agent's trajectory.

Each task's own README documents its scenario, reward layers and contracts in
full.

## Prerequisites

| | | why |
| --- | --- | --- |
| Python | 3.11+ | to run the suites from a checkout; 3.12 in-image |
| Docker | with Compose v2 | two images from one context, joined by a socket volume |
| Harbor | `uv tool install harbor` | runs the task |
| Ollama | optional | the default agent for `task_manager/` runs a local model |

Nothing here needs an API key by default. A hosted model needs its own
credential in `.env` at the repository root; the oracle runs, the test suites
and the offline grader auditors need none.

## Running

From the repository root:

```bash
# Deterministic reference run. No model, no key. Scores exactly 1.0.
harbor run -p ./example_tasks/task_manager -a oracle
harbor run -p ./example_tasks/slack -a oracle

# A model run, with cleanup, writing a job under the task's jobs/harbor/.
./example_tasks/task_manager/run.sh          # local Ollama by default
./example_tasks/slack/run.sh                 # Claude Code through OpenRouter

# Stop a task's Harbor processes, containers and viewer ports.
./example_tasks/task_manager/kill.sh
```

Both tasks ship the same provider-agnostic agent, bound to their own workspace
([`slack/agent/`](slack/agent/), [`task_manager/agent/`](task_manager/agent/)).
Ollama is the default because it costs nothing and needs no account; switching
model is one variable, and the environment stays offline either way because the
loop runs on the host and only the tool calls go into the container.

```bash
MODEL=ollama/gemma4:26b ./example_tasks/task_manager/run.sh
MODEL=openrouter/anthropic/claude-opus-5 ./example_tasks/slack/run.sh

# Or without Harbor at all: a throwaway workspace, then the real verifier.
cd example_tasks/task_manager && PYTHONPATH=environment:. python3 -m agent --local --grade
```

Every model call carries a schema generated from the task's own tool
definitions, so a model cannot name a tool that does not exist or invent an
argument. `qwen3.6:35b` solves `task_manager/` outright at 1.0 and reaches about
0.23 on `slack/`, which is a useful floor rather than a demonstration that the
harder task is easy.

## Tests

Plain scripts, no test runner. Each task's `tests/test.sh` **is** its Harbor
verifier, and running it from a checkout runs the same suites the graded
container does — 17 for `slack/`, 13 for `task_manager/`:

```bash
cd example_tasks/task_manager
export PYTHONPATH=environment:tests:.
sed -n 's/^for suite in \(.*\); do$/\1/p' tests/test.sh | tr ' ' '\n' |
  while read -r s; do python3 "tests/$s.py" >/dev/null || echo "FAILED $s"; done
```

Some checks are authoring-time guards that read `task.toml`, the Dockerfile or
the reference solution. Those files deliberately do not ship, so the guards run
from a checkout and stand aside inside the image.
