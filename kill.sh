#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Forge — Process Killer
# Kills any running backend (uvicorn on :8000) and frontend (Next.js on :3000)
# processes. Safe to run at any time.
# ─────────────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET} $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET} $*"; }
info() { echo -e "  ${BOLD}·${RESET} $*"; }

echo ""
echo -e "${BOLD}Stopping Forge processes...${RESET}"
echo ""

killed_any=false

# ── Kill by port ──────────────────────────────────────────────────────────────
kill_port() {
  local port="$1"
  local label="$2"
  local pids
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    sleep 0.5
    local remaining
    remaining=$(lsof -ti:"$port" 2>/dev/null || true)
    if [[ -n "$remaining" ]]; then
      echo "$remaining" | xargs kill -9 2>/dev/null || true
    fi
    ok "$label (port $port) stopped — PIDs: $pids"
    killed_any=true
  else
    info "Nothing running on port $port ($label)"
  fi
}

kill_port 8000 "Backend (uvicorn)"
kill_port 3000 "Frontend (Next.js)"
kill_port 6379 "Redis"

# ── Stop forge-managed sandbox containers ─────────────────────────────────────
if command -v docker &>/dev/null 2>&1; then
  container_ids=$(docker ps -q --filter "label=forge.managed=true" 2>/dev/null || true)
  if [[ -n "$container_ids" ]]; then
    echo "$container_ids" | xargs docker stop 2>/dev/null || true
    ok "Sandbox containers stopped"
    killed_any=true
  else
    info "No sandbox containers running"
  fi
fi

# ── Kill by process name (belt-and-suspenders) ────────────────────────────────
kill_pattern() {
  local pattern="$1"
  local label="$2"
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    sleep 0.3
    local remaining
    remaining=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [[ -n "$remaining" ]]; then
      echo "$remaining" | xargs kill -9 2>/dev/null || true
    fi
    ok "$label processes terminated — PIDs: $pids"
    killed_any=true
  fi
}

kill_pattern "uvicorn backend.app.main:app" "Stray uvicorn"
kill_pattern "next-server"                  "Stray Next.js server"
kill_pattern "next dev"                     "Stray next dev"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
if [[ "$killed_any" == "true" ]]; then
  echo -e "${GREEN}${BOLD}All Forge processes stopped.${RESET}"
else
  echo -e "${YELLOW}No Forge processes were running.${RESET}"
fi
echo ""
