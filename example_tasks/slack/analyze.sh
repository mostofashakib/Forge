#!/usr/bin/env bash
# Run `harbor analyze` on a trial or job with this repository's credentials.
#
# `harbor analyze` launches a Claude Code agent of its own to read the
# trajectory, and that agent needs the same OpenRouter credentials `run.sh`
# gives the agent under test. Without them the inner agent reports
# "Not logged in", which surfaces as AgentAuthenticationError on the CLI and as
# a bare 500 from the viewer's /summarize endpoint.
#
# The model must also be a real OpenRouter id. Harbor's own default
# (`claude-haiku-4-5`) and the viewer's (`haiku`) are Anthropic tier aliases,
# and OpenRouter rejects both with "is not a valid model ID".
#
# Sonnet 5 rather than Haiku 4.5, which reaches the right conclusions but often
# writes them into its final message instead of the analysis.json the verifier
# looks for -- reported as "Analyze agent did not produce a valid result".
# Override with ANALYZE_MODEL=<openrouter-id>.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
MODEL="${ANALYZE_MODEL:-anthropic/claude-sonnet-5}"

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <trial-or-job-dir> [harbor analyze args...]" >&2
  echo "example: $(basename "$0") $SCRIPT_DIR/jobs/harbor/<job>/<trial>" >&2
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

export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_API_KEY="$OPEN_ROUTER_KEY"
unset ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN
unset OPEN_ROUTER_KEY OPENROUTER_API_KEY

TARGET="$1"
shift

exec harbor analyze "$TARGET" -m "$MODEL" "$@"
