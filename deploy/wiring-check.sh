#!/usr/bin/env bash
# pass, fail, and skip always succeed, so `cond && pass || fail` is a safe if/else.
# shellcheck disable=SC2015
# Test every connection in deploy/README.md's wiring map from the process that
# makes it, and confirm the forbidden connections fail.
#
#   deploy/wiring-check.sh [env-file]      (default: deploy/.env.production)
#
# Prints PASS <id>, FAIL <id>: <reason>, or SKIP <id> (<why>) per row, then a
# count. Exits non-zero on any FAIL. A row is skipped only when its credential is
# still mock_key, FORGE_TAILNET_HOST is unset (W1, W19, X1), or no deploy workflow
# run exists yet (W18). Needs docker with compose, and python3.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$DIR/.env.production}"
[[ -f "$ENV_FILE" ]] || { echo "env file not found: $ENV_FILE"; exit 2; }
COMPOSE=(docker compose -f "$DIR/docker-compose.yml" --env-file "$ENV_FILE")
CURL_IMAGE=curlimages/curl:8.23.0

env_value() { grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//'; }
configured() { local value; value=$(env_value "$1"); [[ -n "$value" && "$value" != "mock_key" ]]; }

TAILNET_HOST=$(env_value FORGE_TAILNET_HOST)
BASE="https://$(env_value FORGE_PUBLIC_HOST)"
TLS=()
[[ "$(env_value FORGE_TLS)" == "internal" ]] && TLS=(-k)   # Caddy's own CA on a test machine
ORIGIN=$(env_value CORS_ORIGINS | cut -d, -f1)
BACKEND_IMAGE="$(env_value FORGE_IMAGE_REPO)/forge-backend:$(env_value FORGE_IMAGE_TAG)"

passed=0; failed=0; skipped=0
pass() { echo "PASS $1"; passed=$((passed + 1)); }
fail() { echo "FAIL $1: $2"; failed=$((failed + 1)); }
skip() { echo "SKIP $1 ($2)"; skipped=$((skipped + 1)); }
check() { local id="$1"; shift; local out; if out=$("$@" 2>&1); then pass "$id"; else fail "$id" "$(tail -n 3 <<<"$out" | tr '\n' ' ')"; fi; }

# curl from the stack's own network namespace, so localhost means the stack's host.
hcurl() { docker run --rm --network host "$CURL_IMAGE" -sS --max-time 15 "${TLS[@]}" "$@"; }
in_service() { local service="$1"; shift; "${COMPOSE[@]}" exec -T "$service" "$@"; }
ws_status() {  # HTTP status of a WebSocket upgrade; 101 means the handshake succeeded
    hcurl --max-time 5 -o /dev/null -w '%{http_code}' --http1.1 -H "Connection: Upgrade" -H "Upgrade: websocket" \
        -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: Zm9yZ2Utd2lyaW5nLWtleQ==" "$1" || true
}
health_field() { in_service api python -c "import json,urllib.request,sys
try: body = urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=10).read()
except urllib.error.HTTPError as exc: body = exc.read()
print(json.loads(body)['checks']['$1'])"; }
age_under() {  # $1: ISO timestamp, $2: max age in seconds
    python3 -c "import sys,datetime as d; t=d.datetime.fromisoformat(sys.argv[1].replace('Z','+00:00')); sys.exit(0 if (d.datetime.now(d.timezone.utc)-t).total_seconds() < int(sys.argv[2]) else 1)" "$1" "$2"
}

# A stand-in sandbox gateway: published on 127.0.0.1 like Forge's gateways.
GATEWAY=""
cleanup() { [[ -n "$GATEWAY" ]] && docker rm -f "$GATEWAY" >/dev/null 2>&1; }
trap cleanup EXIT
GATEWAY=$(docker run -d --rm -p 127.0.0.1::8000 "$BACKEND_IMAGE" python -m http.server 8000 2>/dev/null)
GATEWAY_PORT=$(docker port "$GATEWAY" 8000/tcp 2>/dev/null | head -1 | sed 's/.*://')
sleep 2

# ---------------------------------------------------------------- W rows
if [[ -n "$TAILNET_HOST" ]]; then
    check W1 curl -fsS --max-time 15 -o /dev/null "$BASE/api/health"
else
    skip W1 "FORGE_TAILNET_HOST unset"
fi

w2=$(hcurl "$BASE/" 2>&1 || true)
grep -qi '<html' <<<"$w2" && pass W2 || fail W2 "no app shell at $BASE/: ${w2:0:80}"

w3_health=$(hcurl -o /dev/null -w '%{http_code}' "$BASE/api/health" || true)
w3_ws=$(ws_status "$BASE/api/sandbox/ws/progress/wiring-check")
if [[ "$w3_health" =~ ^(200|503)$ && "$w3_ws" == "101" ]]; then pass W3; else fail W3 "health HTTP $w3_health, websocket HTTP $w3_ws"; fi

w4=$(hcurl -o /dev/null -w '%{http_code}' "$BASE/logs/" || true)
[[ "$w4" == "200" ]] && pass W4 || fail W4 "/logs/ answered HTTP $w4"

w5=$(hcurl -o /dev/null -D - -X OPTIONS -H "Origin: $ORIGIN" -H "Access-Control-Request-Method: GET" "$BASE/api/health" | tr -d '\r' || true)
grep -qi "^access-control-allow-origin: $ORIGIN$" <<<"$w5" && pass W5 || fail W5 "preflight from $ORIGIN not allowed"

check W6 in_service api python -c "import os, redis; redis.from_url(os.environ['REDIS_URL'], socket_timeout=5).ping()"
w7=$(in_service worker celery -A backend.app.worker.celery_app inspect ping --timeout=5 2>&1 || true)
grep -q pong <<<"$w7" && pass W7 || fail W7 "$(tail -n 2 <<<"$w7" | tr '\n' ' ')"

sqlite_rw='import sqlite3; c = sqlite3.connect("/data/forge.db", timeout=10); c.execute("BEGIN IMMEDIATE"); c.execute("SELECT count(*) FROM sandbox_environments").fetchone(); c.rollback()'
w8_api=$(in_service api python -c "$sqlite_rw" 2>&1) && w8_worker=$(in_service worker python -c "$sqlite_rw" 2>&1) \
    && pass W8 || fail W8 "$w8_api $w8_worker"

w9_api=$(in_service api docker version --format '{{.Server.Version}}' 2>&1) \
    && w9_worker=$(in_service worker docker version --format '{{.Server.Version}}' 2>&1) \
    && pass W9 || fail W9 "api: ${w9_api:-} worker: ${w9_worker:-}"

if [[ -n "$GATEWAY_PORT" ]]; then
    check W10 in_service worker python -c "import httpx; from forge.settings import sandbox_url; httpx.get(sandbox_url($GATEWAY_PORT), timeout=5).raise_for_status()"
    w11_reach=$(in_service frontend node -e "fetch('http://' + process.env.FORGE_SANDBOX_HOST + ':$GATEWAY_PORT/').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))" 2>&1 && echo ok)
    w11_proxy=$(hcurl -o /dev/null -w '%{http_code}' "$BASE/api/proxy/wiring-check-missing/ui" || true)
    w11_browser=$(hcurl -s "$BASE/api/sandbox/wiring-check-missing/browser/" || true)
    if [[ "$w11_reach" == *ok && "$w11_proxy" == "503" && "$w11_browser" == *"not found"* ]]; then pass W11
    else fail W11 "frontend reach: ${w11_reach:-failed}, /api/proxy HTTP $w11_proxy, browser proxy: ${w11_browser:0:80}"; fi
else
    fail W10 "could not start the stand-in gateway from $BACKEND_IMAGE"
    fail W11 "could not start the stand-in gateway from $BACKEND_IMAGE"
fi

w12=$(health_field llm 2>&1)
[[ "$w12" == "ok" ]] && pass W12 || fail W12 "llm check: $w12"

if configured BACKUP_S3_SECRET_ACCESS_KEY; then
    newest=$(in_service litestream litestream ltx -config /etc/litestream.yml /data/forge.db 2>&1 | awk 'NR > 1 { print $NF }' | sort | tail -1)
    if [[ -n "$newest" ]] && age_under "$newest" 3600; then pass W13; else fail W13 "newest replica file: ${newest:-none}"; fi
    restic_latest=$(in_service backup restic snapshots --latest 1 --json 2>&1 | python3 -c "import json,sys; s=json.load(sys.stdin); print(s[-1]['time'] if s else '')" 2>/dev/null)
    if [[ -n "$restic_latest" ]] && age_under "$restic_latest" 90000; then pass W14; else fail W14 "latest restic snapshot: ${restic_latest:-none}"; fi
else
    skip W13 "BACKUP_S3_SECRET_ACCESS_KEY is mock_key"
    skip W14 "BACKUP_S3_SECRET_ACCESS_KEY is mock_key"
fi

hc_last_ping() {  # $1: check name. Prints its last ping time.
    curl -fsS --max-time 15 -H "X-Api-Key: $(env_value HC_API_KEY_READONLY)" https://healthchecks.io/api/v3/checks/ \
        | python3 -c "import json,sys; print(next((c['last_ping'] or '') for c in json.load(sys.stdin)['checks'] if c['name'] == sys.argv[1]))" "$1"
}
if configured HC_API_KEY_READONLY && configured FORGE_HC_HEALTH_URL; then
    last=$(hc_last_ping forge-health 2>&1)
    [[ -n "$last" ]] && age_under "$last" 600 && pass W15 || fail W15 "forge-health last ping: ${last:-never}"
else
    skip W15 "HC_API_KEY_READONLY or FORGE_HC_HEALTH_URL is mock_key"
fi
if configured HC_API_KEY_READONLY && configured FORGE_HC_BACKUP_URL; then
    last=$(hc_last_ping forge-backup 2>&1)
    [[ -n "$last" ]] && age_under "$last" 90000 && pass W16 || fail W16 "forge-backup last ping: ${last:-never}"
else
    skip W16 "HC_API_KEY_READONLY or FORGE_HC_BACKUP_URL is mock_key"
fi

w17=$(hcurl "$BASE/logs/api/hosts" 2>&1 || true)
w17_logs=$("${COMPOSE[@]}" logs --no-color --tail 50 dozzle 2>&1)
if grep -qiE "error.*docker|cannot connect" <<<"$w17_logs"; then fail W17 "dozzle cannot reach Docker: $(grep -iE 'error|cannot' <<<"$w17_logs" | tail -1)"
elif [[ "$(hcurl -o /dev/null -w '%{http_code}' "$BASE/logs/")" == "200" && "$w17" != *"502"* ]]; then pass W17
else fail W17 "dozzle answered: ${w17:0:80}"; fi

if command -v gh >/dev/null && [[ -f "$DIR/../.github/workflows/deploy.yml" ]] \
    && run=$(gh run list --workflow deploy.yml --branch main --limit 1 --json conclusion --jq '.[0].conclusion' 2>/dev/null) && [[ -n "$run" ]]; then
    [[ "$run" == "success" ]] && pass W18 || fail W18 "latest deploy run: $run"
else
    skip W18 "no deploy workflow run yet"
fi

if [[ -n "$TAILNET_HOST" ]]; then
    check W19 docker buildx imagetools inspect "$BACKEND_IMAGE"
else
    skip W19 "FORGE_TAILNET_HOST unset"
fi

# ---------------------------------------------------------------- X rows (must fail)
if [[ -n "$TAILNET_HOST" ]]; then
    if grep -qE "^X1 evidence: .*filtered" "$DIR/PROGRESS.md" 2>/dev/null; then pass X1
    else fail X1 "no 'X1 evidence: ... filtered' line in deploy/PROGRESS.md (outside scan, H10)"; fi
else
    skip X1 "FORGE_TAILNET_HOST unset"
fi

x2=$(in_service redis valkey-cli -h 127.0.0.1 ping 2>&1 || true)
[[ "$x2" == *NOAUTH* ]] && pass X2 || fail X2 "Redis without a password answered: $x2"

open=$(docker run --rm --network host "$CURL_IMAGE" sh -c 'netstat -tln 2>/dev/null' \
    | awk '$1 ~ /tcp/ { print $4 }' | grep -E ':(6379|8000|3000|9999|11434|2019)$' | grep -vE '^(127\.0\.0\.1|::1|\[::1\]):' || true)
[[ -z "$open" ]] && pass X3 || fail X3 "listening beyond loopback: $(tr '\n' ' ' <<<"$open")"

x4=""
for service in caddy frontend redis litestream backup heartbeat; do
    id=$("${COMPOSE[@]}" ps -q "$service" 2>/dev/null)
    [[ -z "$id" ]] && continue
    docker inspect --format '{{range .Mounts}}{{.Source}} {{end}}' "$id" | grep -q docker.sock && x4+="$service "
done
[[ -z "$x4" ]] && pass X4 || fail X4 "docker.sock mounted in: $x4"

x5=$(hcurl -o /dev/null -D - -X OPTIONS -H "Origin: https://evil.example" -H "Access-Control-Request-Method: GET" "$BASE/api/health" | tr -d '\r' || true)
grep -qi "^access-control-allow-origin:" <<<"$x5" && fail X5 "https://evil.example was allowed" || pass X5

echo "wiring-check: $passed passed, $failed failed, $skipped skipped"
(( failed == 0 ))
