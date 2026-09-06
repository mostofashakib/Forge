#!/usr/bin/env python3
"""JSON CLI over the TaskHandoverEnvironment training contract.

One command per method, so a trainer in any language can drive the environment
over a pipe. Every command prints JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from task_sim.environment import DEFAULT_MAX_TURNS, TaskHandoverEnvironment
from task_sim.service import DEFAULT_DB_PATH, DEFAULT_SNAPSHOT_PATH


def _instruction(args: argparse.Namespace) -> str:
    if args.instruction_file:
        return Path(args.instruction_file).read_text(encoding="utf-8")
    return args.instruction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "setup", "reset", "setup_state", "seed_session", "prompts",
            "render_prompt", "tools", "step", "state", "history",
        ],
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--session-cookie", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-id", default="tracker_operator")
    parser.add_argument("--prompt-context", default="{}")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--instruction-file", default="")
    parser.add_argument("--metadata", default="{}")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--history-limit", type=int, default=None)
    parser.add_argument("--tool-name", default="")
    parser.add_argument("--input-payload", default="{}")
    args = parser.parse_args()

    environment = TaskHandoverEnvironment(args.db, args.snapshot)
    result: Any
    if args.command == "setup":
        result = environment.setup()
    elif args.command == "setup_state":
        environment.setup_state(seed=args.seed)
        result = environment.setup()
    elif args.command == "reset":
        result = environment.reset(
            session_cookie=args.session_cookie,
            seed=args.seed,
            user_instruction=_instruction(args),
            prompt_id=args.prompt_id,
            prompt_context=json.loads(args.prompt_context),
            metadata=json.loads(args.metadata),
            max_turns=args.max_turns,
        )
    elif args.command == "seed_session":
        result = environment.seed_session(
            session_cookie=args.session_cookie,
            user_instruction=_instruction(args),
            prompt_id=args.prompt_id,
            prompt_context=json.loads(args.prompt_context),
            metadata=json.loads(args.metadata),
            max_turns=args.max_turns,
        )
    elif args.command == "prompts":
        result = environment.prompts()
    elif args.command == "render_prompt":
        result = environment.render_prompt(
            args.prompt_id, _instruction(args), json.loads(args.prompt_context)
        )
    elif args.command == "tools":
        result = environment.tool_definitions()
    elif args.command == "step":
        if not args.tool_name:
            parser.error("step requires --tool-name")
        result = environment.step(
            args.session_cookie, args.tool_name, json.loads(args.input_payload)
        )
    elif args.command == "state":
        result = environment.state(args.session_cookie, history_limit=args.history_limit)
    else:
        result = environment.history(args.session_cookie, limit=args.history_limit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
