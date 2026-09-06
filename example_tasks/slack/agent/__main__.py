"""Run the agent without Harbor.

    # against a throwaway workspace on this machine -- no Docker, no key
    python3 -m agent --local --grade

    # against the compose stack
    python3 -m agent --docker slack-main-1

    # from inside the agent container, over agent.sock
    python3 -m agent

    # any provider, same command
    python3 -m agent --local --model openrouter/anthropic/claude-opus-5

Useful for the question Harbor is a heavy way to ask: can this model do the task
at all. ``--grade`` answers it with the same verifier the graded run uses, so
the number here and the number Harbor reports mean the same thing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shlex
import sys
import tempfile
from pathlib import Path

from agent.backends import InProcessBackend, SubprocessBackend, ToolBackend
from agent.config import DEFAULT_MODEL, AgentConfig
from agent.loop import RunResult, ToolLoop
from agent.prompts import DEFAULT_PROMPT_ID, system_prompt
from agent.providers import ProviderError, provider_names
from agent.trajectory import build_trajectory, write_trajectory

TASK_ROOT = Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m agent",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"provider/model, or a bare model name for the default provider "
        f"(default: {DEFAULT_MODEL}; providers: {', '.join(provider_names())})",
    )
    parser.add_argument("--base-url", default=None, help="override the provider endpoint")
    parser.add_argument("--api-key", default=None, help="override the provider credential")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--deadline-sec", type=float, default=None)
    parser.add_argument("--think", action="store_true", help="let a thinking model reason")
    parser.add_argument("--prompt-id", default=None, help=f"operator role (default: {DEFAULT_PROMPT_ID})")

    where = parser.add_argument_group("where tools run")
    where.add_argument("--local", action="store_true", help="a throwaway workspace in this process")
    where.add_argument("--db", type=Path, default=None, help="with --local, an existing workspace")
    where.add_argument("--docker", metavar="CONTAINER", default=None, help="docker exec into a container")
    where.add_argument("--exec-prefix", default=None, help="arbitrary prefix in front of `slack`")

    parser.add_argument("--instruction-file", type=Path, default=TASK_ROOT / "instruction.md")
    parser.add_argument("--instruction", default=None, help="overrides --instruction-file")
    parser.add_argument("--trajectory", type=Path, default=None, help="write ATIF here")
    parser.add_argument("--state-out", type=Path, default=None, help="with --local, write the state export here")
    parser.add_argument("--grade", action="store_true", help="with --local, score the result")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _backend(args: argparse.Namespace, workspace: Path) -> tuple[ToolBackend, Path | None]:
    if args.local:
        from slack_sim.service import seed_database

        db_path = args.db
        if db_path is None:
            db_path = workspace / "tasks.db"
            seed_database(db_path, workspace / "seed.sql")
        return InProcessBackend(db_path), db_path
    if args.docker:
        return SubprocessBackend(["docker", "exec", "-u", "agent", args.docker]), None
    if args.exec_prefix:
        return SubprocessBackend(shlex.split(args.exec_prefix)), None
    return SubprocessBackend(), None


def _report(result: RunResult, *, quiet: bool) -> None:
    if quiet:
        return
    for turn in result.turns:
        if turn.text.strip():
            print(f"\n[{turn.index}] {turn.text.strip()}")
        for call, envelope in zip(turn.tool_calls, turn.results, strict=False):
            mark = "ok " if envelope.get("ok") else "ERR"
            payload = json.dumps(call.arguments, sort_keys=True)
            print(f"  {mark} {call.name} {payload[:160]}")
            if not envelope.get("ok"):
                error = envelope.get("error") or {}
                print(f"      -> {error.get('code')}: {error.get('message')}")
    print(f"\n{json.dumps(result.summary(), indent=2)}")


def _grade(db_path: Path, state_out: Path | None) -> int:
    from slack_sim.service import export_state
    from verifiers.contracts.acme_migration import build_contract
    from verifiers.episode import Episode
    from verifiers.presets import TieredRewardEngine

    state = export_state(db_path)
    if state_out:
        state_out.parent.mkdir(parents=True, exist_ok=True)
        state_out.write_text(json.dumps(state, indent=2), encoding="utf-8")
    evaluation = TieredRewardEngine.for_preset(build_contract(), None).evaluate(
        Episode.from_state(state)
    )
    print()
    print(evaluation.summary())
    return 0 if evaluation.valid else 2


async def _run(args: argparse.Namespace, workspace: Path) -> int:
    config = AgentConfig.from_env(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_turns=args.max_turns,
        deadline_sec=args.deadline_sec,
        prompt_id=args.prompt_id,
        think=True if args.think else None,
    )
    try:
        provider = config.build_provider()
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backend, db_path = _backend(args, workspace)
    instruction = args.instruction or args.instruction_file.read_text(encoding="utf-8")
    if not args.quiet:
        print(f"{provider.spec} -> {backend.describe()}")

    result = await ToolLoop(
        provider,
        backend,
        system=system_prompt(config.prompt_id),
        max_turns=config.max_turns,
        max_tool_output_chars=config.max_tool_output_chars,
        repeat_limit=config.repeat_limit,
        deadline_sec=config.deadline_sec,
    ).run(instruction)
    _report(result, quiet=args.quiet)

    if args.trajectory:
        write_trajectory(
            args.trajectory,
            build_trajectory(
                result,
                agent_name="workspace-adapter",
                agent_version="1.0.0",
                model_name=provider.spec,
                extra=provider.describe(),
            ),
        )
        if not args.quiet:
            print(f"wrote {args.trajectory}")

    if args.grade or args.state_out:
        if db_path is None:
            print("error: --grade and --state-out need --local", file=sys.stderr)
            return 2
        return _grade(db_path, args.state_out)
    return 0 if result.stop_reason in {"answered", "max_turns"} else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    with tempfile.TemporaryDirectory(prefix="tracker-agent-") as scratch:
        return asyncio.run(_run(args, Path(scratch)))


if __name__ == "__main__":
    raise SystemExit(main())
