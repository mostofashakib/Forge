"""Task Manager simulation driver entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.simulation_driver import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_AGENT_PROVIDER,
    builtin_task_manager_task,
    load_tasks,
    run_task_list,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Task Manager simulations.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tasks", help="Path to a JSON list of Task Manager task definitions.")
    parser.add_argument("--output", default="/tmp/fleet/task_manager_trajectory.json")
    parser.add_argument("--agent-provider", default=DEFAULT_AGENT_PROVIDER)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--real-agent", action="store_true")
    parser.add_argument("--transcript", help="Path for formatted text debug output. Defaults to output path with .txt.")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks, "task_manager") if args.tasks else [builtin_task_manager_task()]
    transcript_path = args.transcript or str(Path(args.output).with_suffix(".txt"))
    run_task_list(
        tasks=tasks,
        seed=args.seed,
        output_path=args.output,
        verbose=not args.quiet,
        agent_provider=args.agent_provider,
        agent_model=args.agent_model,
        real_agent=args.real_agent,
        transcript_path=transcript_path,
    )


if __name__ == "__main__":
    main()
