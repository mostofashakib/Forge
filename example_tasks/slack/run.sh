#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
TASK_PATH="$SCRIPT_DIR"
MODEL="anthropic/claude-opus-4.7"
# Keep one level of directories below JOBS_PATH: each child is a complete
# Harbor job containing config.json/result.json plus its trial directories.
# This is also the directory to pass to `harbor view`. Pointing the viewer at a
# single job directory makes it mistake trial result.json files for job results.
JOBS_PATH="$SCRIPT_DIR/jobs/harbor"
JOB_NAME="claude-opus-4.7__$(date -u +%Y-%m-%d__%H-%M-%S)"

CLEANUP=1
HARBOR_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --no-cleanup) CLEANUP=0 ;;
    *) HARBOR_ARGS+=("$arg") ;;
  esac
done

if [ "$CLEANUP" -eq 1 ]; then
  "$SCRIPT_DIR/kill.sh"
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

# Claude Code uses OpenRouter's Anthropic-compatible API. Harbor's Claude Code
# adapter resolves its credential through ANTHROPIC_API_KEY and forwards the
# configured base URL. Keep the original project variable local.
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_API_KEY="$OPEN_ROUTER_KEY"
unset ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN
unset OPEN_ROUTER_KEY OPENROUTER_API_KEY

exec harbor run \
  -p "$TASK_PATH" \
  -a claude-code \
  -m "$MODEL" \
  --ak reasoning_effort=high \
  --ak max_turns=80 \
  --ak max_budget_usd=25 \
  --n-concurrent 1 \
  --job-name "$JOB_NAME" \
  -o "$JOBS_PATH" \
  ${HARBOR_ARGS[@]+"${HARBOR_ARGS[@]}"}
