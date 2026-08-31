#!/usr/bin/env bash
# Fleet deterministic environments — local validation runner.
# Keeps setup, validation, tests, simulations, and Harbor tasks behind one
# documented entry point. Dependency installation only occurs for `setup`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Agent model for smoke tests and harbor evals. Override with e.g.
#   FLEET_MODEL=anthropic/claude-sonnet-4-5 ./run.sh slack-task-1
MODEL="${FLEET_MODEL:-gemma4:26b}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${BOLD}[fleet]${RESET} $*"; }
ok()   { echo -e "${GREEN}  ✓${RESET} $*"; }
warn() { echo -e "${YELLOW}  ⚠${RESET} $*"; }
err()  { echo -e "${RED}  ✗${RESET} $*"; }

usage() {
  cat <<'EOF'
Usage: ./run.sh [command]

Commands:
  setup         Create .venv and sync the locked dependencies with uv.
  doctor        Check local tools and explain any missing prerequisites.
  validate      Validate task structure and metadata without dependencies.
  test          Run deterministic/unit verifier tests without Ollama.
  smoke         Run smoke tests with the real Ollama agent.
  driver        Run deterministic Slack and Task Manager simulations.
  real-driver   Run simulations with the real Ollama agent.
  all           Run test, driver, and smoke.
  slack-task-1              Run harbor eval for slack_task_1.
  slack-task-2              Run harbor eval for slack_task_2.
  slack-tasks               Run harbor eval for all slack tasks.
  task-manager-task-1       Run harbor eval for task_manager_task_1.
  task-manager-tasks        Run harbor eval for all task manager tasks.
  benchmark                 Run harbor eval for all three tasks.
  kill-harbor-ports
                Kill listeners on Harbor's default port range 8080-8089.
  help          Show this message.

Default: help
EOF
}

setup_environment() {
  if ! command -v uv >/dev/null 2>&1; then
    err "uv is required. Install it first: https://docs.astral.sh/uv/"
    exit 1
  fi

  log "Creating .venv and syncing the lockfile with uv..."
  uv sync --frozen --extra dev --python 3.12
  ok "Python environment ready"
}

require_environment() {
  if [[ ! -x "$ROOT/.venv/bin/python" || ! -x "$ROOT/.venv/bin/harbor" ]]; then
    err "Runtime environment is not ready. Run: ./run.sh setup"
    exit 1
  fi
}

run_doctor() {
  local failed=0
  if command -v uv >/dev/null 2>&1; then
    ok "uv: $(uv --version)"
  else
    err "uv is missing: https://docs.astral.sh/uv/"
    failed=1
  fi
  if command -v docker >/dev/null 2>&1; then
    ok "Docker CLI is available"
  else
    warn "Docker is missing; Harbor tasks require it"
  fi
  if [[ -x "$ROOT/.venv/bin/harbor" ]]; then
    ok "Harbor is available in .venv"
  else
    warn "Harbor is not installed locally; run ./run.sh setup"
  fi
  if command -v ollama >/dev/null 2>&1; then
    ok "Ollama is available for smoke and real-agent runs"
  else
    warn "Ollama is optional and only needed for local-model runs"
  fi
  return "$failed"
}

run_validate() {
  log "Validating example task structure and metadata..."
  python3 "$ROOT/scripts/validate_examples.py"
}

run_test() {
  log "Running deterministic/unit verifier tests..."
  "$ROOT/.venv/bin/python" -m unittest \
    tests.test_task_contracts \
    tests.slack.test_deterministic \
    tests.task_manager.test_deterministic \
    tests.test_simulation_driver \
    tests.slack.test_harbor_verifiers
}

run_smoke() {
  log "Running smoke tests with model ${MODEL}..."
  FLEET_MODEL="$MODEL" "$ROOT/.venv/bin/python" tests/test_smoke_tasks.py
}

run_driver() {
  log "Running deterministic simulation driver..."
  "$ROOT/.venv/bin/python" -m tests.simulation_driver \
    --environment all \
    --seed 1 \
    --output /tmp/fleet/trajectories.json
}

run_real_driver() {
  log "Running real-agent simulation driver with model ${MODEL}..."
  "$ROOT/.venv/bin/python" -m tests.simulation_driver \
    --environment all \
    --real-agent \
    --agent-model "$MODEL" \
    --seed 1 \
    --output /tmp/fleet/real_agent.json
}

run_slack_task() {
  local n="$1"
  log "Running harbor eval for slack_task_${n}..."
  "$ROOT/.venv/bin/harbor" run -p "slack_task_${n}" \
    --agent-import-path fleet.agents.rl_agent:SlackExternalAgent \
    --model "$MODEL"
}

run_task_manager_task() {
  local n="$1"
  log "Running harbor eval for task_manager_task_${n}..."
  "$ROOT/.venv/bin/harbor" run -p "task_manager_task_${n}" \
    --agent-import-path fleet.agents.rl_agent:TaskManagerExternalAgent \
    --model "$MODEL"
}

kill_harbor_ports() {
  local start_port="${1:-8080}"
  local end_port="${2:-8089}"
  local grace_seconds="${GRACE_SECONDS:-2}"

  if ! command -v lsof >/dev/null 2>&1; then
    err "lsof is required to inspect listening ports."
    exit 1
  fi

  if ! [[ "$start_port" =~ ^[0-9]+$ && "$end_port" =~ ^[0-9]+$ ]]; then
    err "Usage: ./run.sh kill-harbor-ports [start_port] [end_port]"
    exit 1
  fi

  if (( start_port > end_port )); then
    err "start_port must be less than or equal to end_port."
    exit 1
  fi

  log "Freeing Harbor port range ${start_port}-${end_port}..."

  local pids=()
  local port
  for ((port=start_port; port<=end_port; port++)); do
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      pids+=("$pid")
      log "Port ${port} is used by PID ${pid}"
      lsof -nP -iTCP:"$port" -sTCP:LISTEN || true
    done < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  done

  if (( ${#pids[@]} == 0 )); then
    ok "No listeners found."
    return 0
  fi

  local unique_pids
  unique_pids="$(printf '%s\n' "${pids[@]}" | sort -u)"

  log "Sending SIGTERM to:"
  printf '  %s\n' $unique_pids
  kill $unique_pids 2>/dev/null || true

  sleep "$grace_seconds"

  local remaining=()
  local pid
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      remaining+=("$pid")
    fi
  done <<< "$unique_pids"

  if (( ${#remaining[@]} > 0 )); then
    log "Sending SIGKILL to remaining PIDs:"
    printf '  %s\n' "${remaining[@]}"
    kill -9 "${remaining[@]}" 2>/dev/null || true
  fi

  ok "Done"
}

command="${1:-help}"

case "$command" in
  help|-h|--help)
    usage
    exit 0
    ;;
  setup)
    setup_environment
    ;;
  doctor)
    run_doctor
    ;;
  validate)
    run_validate
    ;;
  test|smoke|driver|real-driver|all|slack-task-1|slack-task-2|slack-tasks|task-manager-task-1|task-manager-tasks|benchmark)
    require_environment
    ;;
  kill-harbor-ports)
    ;;
  *)
    err "Unknown command: $command"
    usage
    exit 1
    ;;
esac

case "$command" in
  setup|doctor|validate)
    ;;
  test)
    run_test
    ;;
  smoke)
    run_smoke
    ;;
  driver)
    run_driver
    ;;
  real-driver)
    run_real_driver
    ;;
  all)
    run_test
    run_driver
    run_smoke
    ;;
  slack-task-1)
    run_slack_task 1
    ;;
  slack-task-2)
    run_slack_task 2
    ;;
  slack-tasks)
    run_slack_task 1
    run_slack_task 2
    ;;
  task-manager-task-1)
    run_task_manager_task 1
    ;;
  task-manager-tasks)
    run_task_manager_task 1
    ;;
  benchmark)
    run_slack_task 1
    run_slack_task 2
    run_task_manager_task 1
    ;;
  kill-harbor-ports)
    kill_harbor_ports "${2:-}" "${3:-}"
    ;;
esac

ok "Done"
