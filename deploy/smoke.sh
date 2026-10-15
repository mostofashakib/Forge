#!/usr/bin/env bash
# One full user flow through the front door, the way a teammate's browser uses Forge:
#   health -> live progress socket -> premade environment -> sandbox page through
#   the proxy -> 3-episode run -> screened trajectory export.
#
#   deploy/smoke.sh https://forge.<tailnet>.ts.net
#   deploy/smoke.sh https://localhost --insecure    (Caddy's internal CA on a test machine)
#
# Cleans up what it created. Exits non-zero at the first failed step.
# Needs bash, curl, and python3.
set -euo pipefail

BASE="${1:?usage: smoke.sh <base-url> [--insecure]}"
BASE="${BASE%/}"
CURL=(curl -sS --max-time 30)
if [[ "${2:-}" == "--insecure" ]]; then CURL+=(-k); fi

ENV_NAME="smoke-$(date +%s)"
STARTED=$(date +%s)
CREATED=""

step() { printf '[smoke] %-34s' "$1"; STEP_START=$(date +%s); }
pass() { echo "ok ($(( $(date +%s) - STEP_START ))s)${1:+  $1}"; }
fail() { echo "FAILED: $1"; exit 1; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

cleanup() {
    if [[ -n "$CREATED" ]]; then
        "${CURL[@]}" -o /dev/null -X DELETE "$BASE/api/sandbox/$ENV_NAME" || true
    fi
}
trap cleanup EXIT

wait_for() {  # $1: url, $2: python expression on the JSON body, $3: done values, $4: timeout seconds
    local deadline=$(( $(date +%s) + $4 )) value
    while (( $(date +%s) < deadline )); do
        value=$("${CURL[@]}" "$1" | json "$2" 2>/dev/null || true)
        if [[ " $3 " == *" $value "* ]]; then echo "$value"; return 0; fi
        sleep 5
    done
    echo "${value:-no answer}"
    return 1
}

step "health"
body=$("${CURL[@]}" -w '\n%{http_code}' "$BASE/api/health") || fail "no answer from $BASE/api/health"
code=${body##*$'\n'}
[[ "$code" == "200" ]] || fail "HTTP $code: ${body%$'\n'*}"
pass "$(echo "${body%$'\n'*}" | json '" ".join(f"{k}={v}" for k, v in d["checks"].items())')"

step "progress websocket handshake"
ws_url="$BASE/api/sandbox/ws/progress/$ENV_NAME"
status=$("${CURL[@]}" --max-time 5 -o /dev/null -w '%{http_code}' --http1.1 \
    -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" \
    -H "Sec-WebSocket-Key: Zm9yZ2Utc21va2Uta2V5IQ==" "$ws_url" 2>/dev/null || true)  # curl times out after the 101, by design
[[ "$status" == "101" ]] || fail "expected 101 Switching Protocols from $ws_url, got $status"
pass

step "premade environment starts"
"${CURL[@]}" -f -o /dev/null -X POST "$BASE/api/sandbox/" -H 'content-type: application/json' \
    -d "{\"env_name\":\"$ENV_NAME\",\"env_type\":\"premade:gmail\"}" || fail "POST /api/sandbox/ refused"
CREATED=1
state=$(wait_for "$BASE/api/sandbox/$ENV_NAME" 'd["status"]' "running failed error" 900) \
    || fail "still '$state' after 15 minutes"
[[ "$state" == "running" ]] || fail "build ended '$state'"
pass "$ENV_NAME"

step "sandbox page through the proxy"
page=$("${CURL[@]}" -w '\n%{http_code}' "$BASE/api/proxy/$ENV_NAME/ui") || fail "no answer from the proxy"
[[ "${page##*$'\n'}" == "200" && "$page" == *"<title>"* ]] || fail "proxy returned HTTP ${page##*$'\n'}"
pass

step "3-episode run"
run_id=$("${CURL[@]}" -f -X POST "$BASE/api/sandbox/$ENV_NAME/agent-runs" -H 'content-type: application/json' \
    -d '{"agent_id":"random","objective":"Archive the newest email","num_episodes":3,"max_steps":5}' \
    | json 'd["id"]') || fail "could not start the run"
state=$(wait_for "$BASE/api/sandbox/$ENV_NAME/agent-runs/$run_id" 'd["status"]' "completed failed" 600) \
    || fail "run still '$state' after 10 minutes"
[[ "$state" == "completed" ]] || fail "run ended '$state'"
rewards=$("${CURL[@]}" "$BASE/api/sandbox/$ENV_NAME/agent-runs/$run_id/episodes" \
    | json '" ".join(str(e["total_reward"]) for e in d if e["status"] == "completed" and e["total_reward"] is not None)')
[[ $(wc -w <<<"$rewards") -eq 3 ]] || fail "expected 3 rewarded episodes, got: ${rewards:-none}"
pass "rewards $rewards"

step "screened trajectory export"
# The safety gate screens every line. A misconfigured gate answers 503.
export_file=$(mktemp)
code=$("${CURL[@]}" --max-time 600 -o "$export_file" -w '%{http_code}' \
    "$BASE/api/sandbox/$ENV_NAME/agent-runs/$run_id/export")
lines=$(wc -l <"$export_file" | tr -d ' ')
rm -f "$export_file"
[[ "$code" == "200" ]] || fail "export answered HTTP $code"
(( lines > 0 )) || fail "export was empty"
pass "$lines screened lines"

echo "[smoke] passed in $(( $(date +%s) - STARTED ))s"
