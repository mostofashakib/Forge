#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Forge — Local Development Runner
# Kills any existing processes on :8000/:3000/:6379, then starts all services.
# Ctrl+C gracefully stops everything.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT/frontend"
VENV_DIR="$ROOT/.venv"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${BOLD}[forge]${RESET} $*"; }
ok()   { echo -e "${GREEN}  ✓${RESET} $*"; }
warn() { echo -e "${YELLOW}  ⚠${RESET} $*"; }
err()  { echo -e "${RED}  ✗${RESET} $*"; }

# ── Prefixed log streams ──────────────────────────────────────────────────────
backend_log()  { while IFS= read -r line; do echo -e "${CYAN}[backend] ${RESET}$line"; done; }
frontend_log() { while IFS= read -r line; do echo -e "${GREEN}[frontend]${RESET} $line"; done; }
redis_log()    { while IFS= read -r line; do echo -e "${RED}[redis]   ${RESET}$line"; done; }

# ── Kill a port — uses nc for a fast check, lsof only to get the PIDs ────────
kill_port() {
  local port="$1" label="$2"
  if ! nc -z 127.0.0.1 "$port" 2>/dev/null; then return; fi
  local pids
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    sleep 0.4
    pids=$(lsof -ti:"$port" 2>/dev/null || true)
    [[ -n "$pids" ]] && echo "$pids" | xargs kill -9 2>/dev/null || true
    ok "Cleared $label (port $port)"
  fi
}

# ── Cleanup on Ctrl+C / TERM ─────────────────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""
CELERY_PID=""
REDIS_PID=""
REDIS_TAIL_PID=""

cleanup() {
  echo ""
  log "Shutting down..."
  [[ -n "$BACKEND_PID" ]]    && kill "$BACKEND_PID"    2>/dev/null || true
  [[ -n "$FRONTEND_PID" ]]   && kill "$FRONTEND_PID"   2>/dev/null || true
  [[ -n "$CELERY_PID" ]]     && kill "$CELERY_PID"     2>/dev/null || true
  [[ -n "$REDIS_PID" ]]      && kill "$REDIS_PID"      2>/dev/null || true
  [[ -n "$REDIS_TAIL_PID" ]] && kill "$REDIS_TAIL_PID" 2>/dev/null || true
  if command -v docker &>/dev/null 2>&1; then
    container_ids=$(docker ps -q --filter "label=forge.managed=true" 2>/dev/null || true)
    [[ -n "$container_ids" ]] && echo "$container_ids" | xargs docker stop 2>/dev/null || true
  fi
  exit 0
}
trap cleanup INT TERM

# ── Kill any already-running instances ───────────────────────────────────────
log "Clearing any running processes..."
kill_port 8000 "backend"
kill_port 3000 "frontend"
kill_port 6379 "redis"
pgrep -f "uvicorn.*backend"           2>/dev/null | xargs kill -9 2>/dev/null || true
pgrep -f "next-server"                2>/dev/null | xargs kill -9 2>/dev/null || true
pgrep -f "next dev"                   2>/dev/null | xargs kill -9 2>/dev/null || true
pgrep -f "celery.*worker.*celery_app" 2>/dev/null | xargs kill -9 2>/dev/null || true
if command -v docker &>/dev/null 2>&1; then
  container_ids=$(docker ps -q --filter "label=forge.managed=true" 2>/dev/null || true)
  [[ -n "$container_ids" ]] && echo "$container_ids" | xargs docker stop 2>/dev/null || true
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -f "$VENV_DIR/bin/uvicorn" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    warn "No .venv found — creating one now..."
    python3 -m venv "$VENV_DIR"
  else
    warn ".venv exists but packages are missing — installing now..."
  fi
  "$VENV_DIR/bin/pip" install -q --upgrade pip
  "$VENV_DIR/bin/pip" install -q -e "$ROOT[dev]"
  ok "Virtual environment ready"
fi

ok "Virtual environment ready ($("$VENV_DIR/bin/python" --version))"

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  warn "node_modules missing — running npm install..."
  npm --prefix "$FRONTEND_DIR" install --silent
fi

# ── Redis ─────────────────────────────────────────────────────────────────────
if ! command -v redis-server &>/dev/null; then
  err "redis-server not found. Install it: brew install redis"
  exit 1
fi

REDIS_LOG="/tmp/forge-redis.log"
log "Starting Redis on port 6379"
redis-server --bind 127.0.0.1 --save "" --appendonly no \
  --daemonize no --loglevel notice \
  >"$REDIS_LOG" 2>&1 &
REDIS_PID=$!

# Tail the log file for colored output in the terminal
tail -f "$REDIS_LOG" 2>/dev/null | redis_log &
REDIS_TAIL_PID=$!

# Wait for Redis to report ready — grep on the log is reliable; redis-cli ping is not
for i in $(seq 1 30); do
  if grep -q "Ready to accept connections" "$REDIS_LOG" 2>/dev/null; then break; fi
  sleep 0.2
done
if ! grep -q "Ready to accept connections" "$REDIS_LOG" 2>/dev/null; then
  err "Redis failed to start — check $REDIS_LOG"
  exit 1
fi
ok "Redis ready (pid $REDIS_PID)"

# ── Start backend ─────────────────────────────────────────────────────────────
log "Starting backend on ${CYAN}http://localhost:8000${RESET}"
(
  cd "$ROOT"
  if [[ -f "backend/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source backend/.env
    set +a
  else
    warn "backend/.env not found — LLM calls will fail without API keys"
  fi
  # --reload watches CWD by default; restrict it to backend/forge source so
  # that worker output (generated_envs/), git worktrees, build caches, and
  # the frontend don't trigger a backend restart and tear down WebSockets
  # mid-build.
  "$VENV_DIR/bin/uvicorn" backend.app.main:app \
    --host "${FORGE_HOST:-127.0.0.1}" --port 8000 --reload \
    --reload-dir backend --reload-dir forge \
    --reload-include "*.py" \
    --reload-exclude "generated_envs/*" \
    --reload-exclude ".worktrees/*" \
    --reload-exclude "**/__pycache__/*" \
    --reload-exclude ".pytest_cache/*" \
    --reload-exclude ".ruff_cache/*" \
    --reload-exclude ".mypy_cache/*" \
    --reload-exclude "*.db" \
    --reload-exclude "*.log" 2>&1
) | backend_log &
BACKEND_PID=$!

sleep 2

# ── Start Celery worker ───────────────────────────────────────────────────────
CELERY_LOG="/tmp/forge-celery.log"
log "Starting Celery worker (quiet; errors: $CELERY_LOG)"
(
  cd "$ROOT"
  if [[ -f "backend/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source backend/.env
    set +a
  fi
  # Worker chatter obscures the interactive backend/frontend output. Keep the
  # terminal quiet while retaining actionable failures for local diagnosis.
  "$VENV_DIR/bin/celery" -A backend.app.worker.celery_app worker \
    --loglevel=ERROR --concurrency=2 --without-gossip --without-mingle \
    >"$CELERY_LOG" 2>&1
) &
CELERY_PID=$!

# ── Start frontend ────────────────────────────────────────────────────────────
log "Starting frontend on ${GREEN}http://localhost:3000${RESET}"
(
  cd "$FRONTEND_DIR"
  NODE_OPTIONS="--no-deprecation" npm run dev 2>&1
) | frontend_log &
FRONTEND_PID=$!

# ── Ready ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}Forge is running${RESET}"
echo -e "  ${CYAN}Backend${RESET}  → http://localhost:8000"
echo -e "  ${CYAN}API Docs${RESET} → http://localhost:8000/docs"
echo -e "  ${GREEN}Frontend${RESET} → http://localhost:3000"
echo -e "  ${YELLOW}Celery${RESET}   → worker (quiet, concurrency=2)"
echo -e "  ${RED}Redis${RESET}    → localhost:6379"
echo -e "  Press ${BOLD}Ctrl+C${RESET} to stop all"
echo ""

# ── Wait ─────────────────────────────────────────────────────────────────────
wait "$BACKEND_PID" "$FRONTEND_PID" "$CELERY_PID" "$REDIS_PID"
