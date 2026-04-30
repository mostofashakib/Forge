#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Forge — Local Development Runner
# Clears any existing processes on :8000/:3000, then starts backend + frontend.
# Ctrl+C gracefully stops both.
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

# ── Kill a port (TERM then SIGKILL) ──────────────────────────────────────────
kill_port() {
  local port="$1" label="$2"
  local pids
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    sleep 0.4
    local remaining
    remaining=$(lsof -ti:"$port" 2>/dev/null || true)
    [[ -n "$remaining" ]] && echo "$remaining" | xargs kill -9 2>/dev/null || true
    ok "Cleared $label (port $port)"
  fi
}

# ── Cleanup on Ctrl+C / TERM ─────────────────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  log "Shutting down..."
  [[ -n "$BACKEND_PID" ]]  && kill "$BACKEND_PID"  2>/dev/null && ok "Backend stopped"
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null && ok "Frontend stopped"
  kill_port 8000 "backend"
  kill_port 3000 "frontend"
  exit 0
}
trap cleanup INT TERM

# ── Clear any already-running instances ──────────────────────────────────────
log "Clearing any running processes..."
kill_port 8000 "backend"
kill_port 3000 "frontend"
pgrep -f "uvicorn main:app" 2>/dev/null | xargs kill -9 2>/dev/null || true
pgrep -f "next-server"      2>/dev/null | xargs kill -9 2>/dev/null || true
pgrep -f "next dev"         2>/dev/null | xargs kill -9 2>/dev/null || true

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

# ── Start backend ─────────────────────────────────────────────────────────────
log "Starting backend on ${CYAN}http://localhost:8000${RESET}"
(
  cd "$ROOT"
  "$VENV_DIR/bin/uvicorn" backend.app.main:app --host 0.0.0.0 --port 8000 --reload 2>&1
) | backend_log &
BACKEND_PID=$!

sleep 2

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
echo -e "  Press ${BOLD}Ctrl+C${RESET} to stop both"
echo ""

# ── Wait ─────────────────────────────────────────────────────────────────────
wait $BACKEND_PID $FRONTEND_PID
