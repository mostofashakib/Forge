#!/usr/bin/env bash
# Run the task under Harbor.
#
# The default agent is this task's own (`agent/`), driving a local Ollama model:
# no account, no key, no per-token cost, and the model call happens on this
# machine rather than inside the environment. Point it somewhere else with one
# variable -- the provider is the head of the model spec:
#
#   ./run.sh                                              # ollama/qwen3.6:35b
#   MODEL=ollama/gemma4:26b ./run.sh
#   MODEL=openrouter/anthropic/claude-opus-5 ./run.sh     # needs OPEN_ROUTER_KEY
#   MODEL=anthropic/claude-opus-5 ./run.sh                # needs ANTHROPIC_API_KEY
#   MODEL=openai/gpt-5 ./run.sh                           # needs OPENAI_API_KEY
#
# AGENT switches the harness rather than the model. This task is rated hard --
# staged reviews, evidence that only appears after the action that provokes it,
# and a rehearsal that invalidates readiness already established -- so a full
# coding agent is the honest comparison for it:
#
#   AGENT=claude-code MODEL=anthropic/claude-opus-4.7 ./run.sh
#
# Any further arguments are passed through to `harbor run`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The repository root, two levels up: this task lives under example_tasks/. It
# was one level when the task was vendored in from its own repository, which
# pointed .env at example_tasks/ and found nothing there.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
TASK_PATH="$SCRIPT_DIR"
AGENT="${AGENT:-agent.harbor_agent:WorkspaceAgent}"
MODEL="${MODEL:-ollama/qwen3.6:35b}"
# Keep one level of directories below JOBS_PATH: each child is a complete
# Harbor job containing config.json/result.json plus its trial directories.
# This is also the directory to pass to `harbor view`. Pointing the viewer at a
# single job directory makes it mistake trial result.json files for job results.
JOBS_PATH="$SCRIPT_DIR/jobs/harbor"
JOB_NAME="$(printf '%s' "$MODEL" | tr '/:' '__')__$(date -u +%Y-%m-%d__%H-%M-%S)"

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

# `agent.harbor_agent:WorkspaceAgent` is an import path, and Harbor resolves it
# against the interpreter's own path rather than the task directory. The world's
# package comes along for the ride so the agent can read the named operator
# prompts out of the environment's contract instead of restating them.
export PYTHONPATH="$TASK_PATH:$TASK_PATH/environment${PYTHONPATH:+:$PYTHONPATH}"

# Credentials are needed only by the providers that have accounts. The default
# is Ollama, which has none, so a missing .env is not an error until something
# asks for it.
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

case "$MODEL" in
  ollama/*|ollama) NEEDS_KEY="" ;;
  openrouter/*)    NEEDS_KEY="OPEN_ROUTER_KEY" ;;
  anthropic/*)     NEEDS_KEY="ANTHROPIC_API_KEY" ;;
  openai/*)        NEEDS_KEY="OPENAI_API_KEY" ;;
  *)               NEEDS_KEY="" ;;
esac

# The Claude Code harness reaches its model through Anthropic's variable names,
# so it needs the key re-exported under those regardless of which service is
# actually behind them. The adapter agent does not: it reads each provider's own
# variable, and the one for the chosen provider is the only one it looks at.
if [[ "$AGENT" == "claude-code" ]]; then
  NEEDS_KEY="OPEN_ROUTER_KEY"
fi

if [[ -n "$NEEDS_KEY" && -z "${!NEEDS_KEY:-}" ]]; then
  echo "$MODEL needs $NEEDS_KEY, which is missing or empty in $ENV_FILE" >&2
  echo "Run the default local model instead with: $(basename "$0")" >&2
  exit 1
fi

AGENT_ARGS=()
if [[ "$AGENT" == "claude-code" ]]; then
  export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
  export ANTHROPIC_API_KEY="$OPEN_ROUTER_KEY"
  unset ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN
  unset OPEN_ROUTER_KEY OPENROUTER_API_KEY
  AGENT_ARGS=(--ak reasoning_effort=high --ak max_turns=80 --ak max_budget_usd=25)
fi

exec harbor run \
  -p "$TASK_PATH" \
  -a "$AGENT" \
  -m "$MODEL" \
  ${AGENT_ARGS[@]+"${AGENT_ARGS[@]}"} \
  --n-concurrent 1 \
  --job-name "$JOB_NAME" \
  -o "$JOBS_PATH" \
  ${HARBOR_ARGS[@]+"${HARBOR_ARGS[@]}"}
