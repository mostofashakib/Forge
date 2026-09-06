#!/usr/bin/env bash
# Audit the *grader* over already-graded trials, using this repository's
# OpenRouter credentials.
#
# This never touches a reward. `tools/grader_audit.py` reads episodes Harbor
# has already scored, asks a cheap model whether the handful of word-matching
# rules reached the right conclusion about the text they read, and prints the
# disagreements as suspected contract bugs. The graded image does not contain
# this tool (see .dockerignore) and the verifier package cannot import it.
#
# Sonnet 5 by default: on the trials this was built against, Haiku 4.5 missed
# one of the three known false positives and Sonnet found all three. The
# parser rejects any answer it cannot read, so an outage fails loudly rather
# than becoming a false clean bill of health. Override with AUDIT_MODEL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
TASK_DIR="$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <trial-dir>... [--offline] [--json out.json]" >&2
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${OPEN_ROUTER_KEY:-}" ]]; then
  echo "OPEN_ROUTER_KEY is missing or empty in $ENV_FILE" >&2
  exit 1
fi

exec python3 "$TASK_DIR/tools/grader_audit.py" "$@"
