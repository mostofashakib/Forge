"""Privileged workspace operations for image build and grading.

Reaches the server over ``admin.sock``, which is mode 0600 and owned by the
server's uid. The agent runs unprivileged and cannot open it. Nothing on the
agent's PATH invokes this module.
"""

from __future__ import annotations

import argparse
import json

from task_sim.protocol import ADMIN_SOCKET, request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tasks-admin")
    parser.add_argument("op", choices=["export_state", "seed", "teardown", "ping"])
    parser.add_argument("--socket", default=ADMIN_SOCKET)
    parser.add_argument("--out", default=None, help="Write the result JSON to this path.")
    args = parser.parse_args(argv)

    response = request(args.socket, {"op": args.op}, timeout=120.0)
    payload = json.dumps(response, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        print(payload)
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
