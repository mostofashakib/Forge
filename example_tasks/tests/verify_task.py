"""Smoke-test CLI for running Harbor task verifier specs."""

from __future__ import annotations

import argparse

from fleet.verifiers.rewardkit_checks import evaluate_verifier, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--reward", default="/logs/verifier/reward.txt")
    parser.add_argument("--report", default="/logs/verifier/report.json")
    parser.add_argument("--workspace", default=None, help="Directory holding the service SQLite database.")
    args = parser.parse_args()

    report = evaluate_verifier(args.trajectory, args.spec, args.workspace)
    write_report(report, args.report, args.reward)


if __name__ == "__main__":
    main()
