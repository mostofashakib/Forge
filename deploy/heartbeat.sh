#!/bin/sh
# Every 5 minutes, report Forge's health to Healthchecks.io: the forge-health ping
# URL when /api/health answers 200 and the data volume is under 85% full,
# otherwise /fail with the health body or the disk usage. A missed ping (the api
# or this container is down) raises the same alert after the grace period.
set -u

DISK_LIMIT=85

while :; do
    rm -f /tmp/health.json
    code=$(curl -s -o /tmp/health.json -w '%{http_code}' --max-time 15 http://127.0.0.1:8000/api/health)
    disk=$(df -P /data | awk 'NR == 2 { sub("%", "", $5); print $5 }')
    if [ "$code" = "200" ] && [ "$disk" -lt "$DISK_LIMIT" ]; then
        suffix=""
        message="ok, data volume ${disk}% full"
    else
        suffix="/fail"
        message="health HTTP ${code}: $(head -c 2000 /tmp/health.json 2>/dev/null); data volume ${disk}% full"
    fi
    case "${FORGE_HC_HEALTH_URL:-}" in
        "" | mock_key) echo "[heartbeat] ${message} (FORGE_HC_HEALTH_URL is not configured, so nothing was reported)" ;;
        *) curl -fsS -m 10 --retry 3 -o /dev/null --data-raw "$message" "${FORGE_HC_HEALTH_URL}${suffix}" \
               && echo "[heartbeat] reported${suffix:- ok}: ${message}" \
               || echo "[heartbeat] could not reach Healthchecks.io" ;;
    esac
    sleep 300
done
