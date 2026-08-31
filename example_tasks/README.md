# Forge Example RL Tasks

This Forge module contains deterministic reinforcement-learning environments for testing whether an agent can inspect state, use tools, preserve invariants, and return a verifiable answer.

Forge includes two SQLite-backed example environment families and three Harbor tasks:

| Task | Environment | Core challenge |
| --- | --- | --- |
| `slack_task_1` | Slack | Route messages by entity type, edit the correct look-alike reply, and derive answers from thread state |
| `slack_task_2` | Slack | Keep a same-named channel and group distinct across a multi-step workflow |
| `task_manager_task_1` | Task Manager | Apply conditional reassignment rules without changing unrelated tasks or labels |

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for Python and lockfile management
- Docker for Harbor task environments
- Ollama only for smoke tests and real local-model runs

The project uses `pyproject.toml` and the committed `uv.lock`. The runner never installs dependencies implicitly; installation happens only when you call `setup`.

## Getting started

Run these commands from this directory:

```bash
./run.sh doctor
./run.sh validate
./run.sh setup
./run.sh test
```

The commands serve different purposes:

- `doctor` reports which optional tools are available.
- `validate` checks task structure, metadata, documentation contracts, and Python syntax without installing dependencies.
- `setup` creates `.venv` and installs the exact versions recorded in `uv.lock`.
- `test` runs the deterministic verifier test suite.

Only `setup` installs dependencies. All other commands expect the environment they need to already exist.

## Command reference

```text
./run.sh setup                 Create .venv and sync uv.lock
./run.sh doctor                Check uv, Docker, Harbor, and Ollama
./run.sh validate              Run dependency-free example-task checks
./run.sh test                  Run deterministic/unit verifier tests
./run.sh smoke                 Run live Ollama smoke tests
./run.sh driver                Run deterministic scripted simulations
./run.sh real-driver           Run simulations with Ollama
./run.sh all                   Run test + driver + smoke
./run.sh slack-task-1          Evaluate slack_task_1 with Harbor
./run.sh slack-task-2          Evaluate slack_task_2 with Harbor
./run.sh slack-tasks           Evaluate both Slack tasks
./run.sh task-manager-task-1   Evaluate task_manager_task_1 with Harbor
./run.sh benchmark             Evaluate all three tasks
./run.sh kill-harbor-ports     Free Harbor ports 8080-8089
```

To use a different Ollama model:

```bash
FLEET_MODEL=gemma4:26b ./run.sh slack-task-1
```

Harbor builds each Docker environment from the task's `docker-compose.yaml`, so no separate image-build step is needed. Evaluation output is written to `jobs/<timestamp>/`.

## Task anatomy

Each task directory is self-contained at the Harbor boundary:

```text
task_name/
├── README.md                 Task-specific guide
├── instruction.md            Prompt presented to the agent
├── task.toml                 Harbor resources and artifact contract
├── environment/              Thin wrapper and environment image
├── solution/solve.sh         Deterministic reference solution
└── tests/                    Verifier image and layered checks
```

Reusable implementation lives in `fleet/`:

- `fleet.environments`: deterministic SQLite services, schemas, and seed data
- `fleet.agents`: external agent adapters and ATIF trajectory recording
- `fleet.verifiers`: state, invariant, trajectory, and negative checks

Every episode starts from the same seed. IDs and virtual timestamps are deterministic. State snapshots are recorded in ATIF v1.7 trajectories, and each verifier criterion reports its own pass or fail result.

## Running simulations directly

After running `./run.sh setup`, launch deterministic simulations directly with the managed interpreter:

```bash
.venv/bin/python -m tests.simulation_driver \
  --environment all \
  --seed 1 \
  --output /tmp/fleet/trajectories.json
```

Use `--real-agent --agent-model <model>` to run with Ollama. Add `--quiet` to hide console output while still saving the JSON trajectory and text transcript.

## Package management

Use `uv` for all dependency changes:

```bash
uv add <package>
uv add --optional dev <package>
uv lock
```

Commit both `pyproject.toml` and `uv.lock` whenever dependencies change. Do not commit `.venv`, model files, generated databases, or secrets.

## Adding an example task

1. Copy the closest task shape and give it a unique `forge/<task-name>` in `task.toml`.
2. Keep the environment wrapper thin; reusable behavior belongs in `fleet/`.
3. Write an instruction with explicit entity types and an exact final-answer contract.
4. Add a deterministic reference solution and layered verifier criteria.
5. Add the task to `scripts/validate_examples.py` and `run.sh`.
6. Run `./run.sh validate`, then `./run.sh test` after setup.

The verifier should prove the requested state change, detect unrelated mutations, verify required action ordering when order matters, and reject prohibited tools.
