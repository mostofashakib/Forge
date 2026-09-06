#!/usr/bin/env bash
# Start the Harbor viewer with this repository's credentials.
#
# The viewer runs `harbor analyze` in-process for its Summarize buttons, so the
# server needs the OpenRouter credentials in its own environment -- a viewer
# started without them answers /summarize with a 500 whose cause is the inner
# agent reporting "Not logged in".
#
# Credentials are only half of it. The viewer's model dropdown offers Anthropic
# tier aliases (haiku / sonnet / opus) and OpenRouter accepts none of them, so
# the Summarize button still fails after this. Use the sibling analyze.sh instead and
# read the result in the viewer: Harbor writes the analysis into a job of its
# own, and the Analysis tab of the analysed trial picks it up.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
JOBS_PATH="$SCRIPT_DIR/jobs/harbor"

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

export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_API_KEY="$OPEN_ROUTER_KEY"
unset ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN
unset OPEN_ROUTER_KEY OPENROUTER_API_KEY

exec harbor view "$JOBS_PATH" "$@"
