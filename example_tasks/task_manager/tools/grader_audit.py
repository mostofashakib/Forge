#!/usr/bin/env python3
"""Audit the *grader* over episodes Harbor already scored.

This never touches a reward. It re-reads finished trials, asks whether the
contract's deterministic rules reached the right conclusion about the state they
read, and prints the disagreements as suspected contract bugs.

Two things keep it honest about its own place. It is excluded from the graded
image (see `.dockerignore`) and no module under `verifiers/` may import it, so
the grader it audits cannot depend on it. And `--offline` skips the model
entirely and reports only what the rules did, which is the mode that needs no
credentials and no network.

    python3 tools/grader_audit.py <trial-dir>... [--offline] [--json out.json]

A trial directory is one produced by `harbor run`; the auditor reads the state
export collected from it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))

from verifiers import Episode, TieredRewardEngine  # noqa: E402
from verifiers.contracts.reassignment import build_contract  # noqa: E402

DEFAULT_MODEL = os.environ.get("AUDIT_MODEL", "anthropic/claude-sonnet-5")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# Finding the evidence
# ---------------------------------------------------------------------------


def find_state_export(trial: Path) -> Path | None:
    """The state export inside a finished trial, wherever Harbor filed it."""
    if trial.is_file():
        return trial
    for candidate in (
        trial / "artifacts" / "var" / "lib" / "tasks" / "state-export.json",
        trial / "state-export.json",
    ):
        if candidate.is_file():
            return candidate
    matches = sorted(trial.rglob("state-export.json"))
    return matches[0] if matches else None


def load_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("result", payload)


# ---------------------------------------------------------------------------
# What the rules concluded
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """One deterministic check, and the state it read to decide."""

    name: str
    layer: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "layer": self.layer, "passed": self.passed,
                "detail": self.detail}


@dataclass(frozen=True, slots=True)
class TrialAudit:
    trial: str
    reward: float
    verdicts: tuple[RuleVerdict, ...]
    disagreements: tuple[dict[str, Any], ...] = ()
    model_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial": self.trial,
            "reward": round(self.reward, 6),
            "verdicts": [verdict.as_dict() for verdict in self.verdicts],
            "disagreements": list(self.disagreements),
            "model_error": self.model_error,
        }


def grade(state: dict[str, Any]) -> tuple[float, tuple[RuleVerdict, ...]]:
    evaluation = TieredRewardEngine.for_preset(build_contract()).evaluate(
        Episode.from_state(state)
    )
    verdicts = tuple(
        RuleVerdict(check.name, result.layer, check.passed, check.detail)
        for result in evaluation.results
        for check in result.checks
    )
    return evaluation.reward, verdicts


# ---------------------------------------------------------------------------
# The second opinion
# ---------------------------------------------------------------------------

PROMPT = """\
You are auditing an automated grader, not an agent.

A task asked an operator to hand over a departing teammate's queue in a task
tracker. Below is what the grader's deterministic rules concluded about the
final state of that tracker, and the state each rule read.

For each rule, say whether the conclusion follows from the state it read. You
are looking for rules that reached the wrong conclusion about the text or the
rows in front of them -- not for whether the operator did well.

Answer with a JSON array and nothing else. One object per rule you disagree
with, each with "name", "grader_said" (true or false) and "why". An empty array
means the grader was right about everything.

RULES AND THE STATE THEY READ:
{body}
"""


def ask_model(verdicts: tuple[RuleVerdict, ...], model: str, api_key: str) -> tuple[list[dict], str | None]:
    body = "\n".join(
        f"- {verdict.name} (layer {verdict.layer}) -> "
        f"{'passed' if verdict.passed else 'failed'}; read: "
        f"{json.dumps(verdict.detail, sort_keys=True)[:600]}"
        for verdict in verdicts
    )
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": PROMPT.format(body=body)}],
            "temperature": 0,
        }).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
        return [], f"{type(exc).__name__}: {exc}"

    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # An answer that cannot be read is not a clean bill of health. Reporting
        # it as one is how an outage becomes a false all-clear.
        return [], f"unparseable model answer: {exc}: {text[:200]!r}"
    if not isinstance(parsed, list):
        return [], f"model answered with {type(parsed).__name__}, not a list"
    return [item for item in parsed if isinstance(item, dict)], None


def audit_trial(trial: Path, offline: bool, model: str, api_key: str) -> TrialAudit:
    export = find_state_export(trial)
    if export is None:
        return TrialAudit(str(trial), 0.0, (), (), "no state export found in this trial")
    reward, verdicts = grade(load_state(export))
    if offline:
        return TrialAudit(str(trial), reward, verdicts)
    disagreements, error = ask_model(verdicts, model, api_key)
    return TrialAudit(str(trial), reward, verdicts, tuple(disagreements), error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", type=Path, help="trial directories, or a state export")
    parser.add_argument("--offline", action="store_true",
                        help="report only what the rules did; no model, no network")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json", type=Path, help="write the full audit here")
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPEN_ROUTER_KEY", "")
    if not args.offline and not api_key:
        print("OPEN_ROUTER_KEY is not set; use --offline to audit without a model.",
              file=sys.stderr)
        return 2

    audits = [audit_trial(trial, args.offline, args.model, api_key) for trial in args.trials]
    for audit in audits:
        print(f"\n{audit.trial}")
        print(f"  reward {audit.reward:.3f}, {len(audit.verdicts)} rules")
        failed = [verdict for verdict in audit.verdicts if not verdict.passed]
        print(f"  {len(failed)} failed: {', '.join(v.name for v in failed[:8])}")
        if audit.model_error:
            print(f"  model unavailable: {audit.model_error}")
        for item in audit.disagreements:
            print(f"  SUSPECTED CONTRACT BUG  {item.get('name')}: {item.get('why')}")
        if not args.offline and not audit.disagreements and not audit.model_error:
            print("  no disagreements")

    if args.json:
        args.json.write_text(
            json.dumps([audit.as_dict() for audit in audits], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    # A model that could not be reached is a failed audit, not a clean one.
    return 1 if any(audit.model_error for audit in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
