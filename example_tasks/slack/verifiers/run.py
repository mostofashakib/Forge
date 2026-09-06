"""Grade one episode from the command line.

    python3 -m verifiers.run --state state-export.json
    python3 -m verifiers.run --state state-export.json --preset binary_final_state
    python3 -m verifiers.run --state state-export.json --json out.json

Reads the world's state export -- the same artefact Harbor already collects --
and prints the verdict with every layer, check and penalty that produced it.
Needs `slack_sim` on the path for the answer key: run with
`PYTHONPATH=environment:.` from a checkout, or `/opt/grading:/opt` in the
graded container.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifiers.episode import Episode
from verifiers.presets import REWARD_PRESETS, TieredRewardEngine, load_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path, help="state export JSON")
    parser.add_argument("--preset", choices=sorted(REWARD_PRESETS), help="overrides the config")
    parser.add_argument("--experiment", type=Path, help="config file carrying reward_preset")
    parser.add_argument("--json", type=Path, help="write the full evaluation here")
    parser.add_argument("--contract", default="acme_migration",
                        help="module under verifiers.contracts providing build_contract()")
    args = parser.parse_args(argv)

    preset = args.preset
    if preset is None and args.experiment:
        preset = load_experiment(args.experiment).get("reward_preset")

    module = __import__(f"verifiers.contracts.{args.contract}", fromlist=["build_contract"])
    contract = module.build_contract()

    episode = Episode.from_files(args.state)
    evaluation = TieredRewardEngine.for_preset(contract, preset).evaluate(episode)

    print(evaluation.summary())
    if evaluation.audit and evaluation.audit["findings"]:
        print("  audit:")
        for finding in evaluation.audit["findings"]:
            print(f"    {finding['code']}: {finding['detail']}")
    if args.json:
        args.json.write_text(json.dumps(evaluation.as_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if evaluation.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
