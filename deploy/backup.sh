#!/bin/sh
# Back up generated environments and their exports with restic, then report to
# Healthchecks.io: the forge-backup ping URL on success, /fail with the error tail.
#
#   /backup.sh        back up every 24 hours (the backup service's command)
#   /backup.sh once   back up once and exit (manual runs and drills)
#
# The database is not here: Litestream replicates it continuously.
set -u

LOG=/tmp/backup.log

ping_check() {  # $1: "" or "/fail", $2: message
    case "${FORGE_HC_BACKUP_URL:-}" in
        "" | mock_key) echo "[backup] FORGE_HC_BACKUP_URL is not configured, so nothing was reported" ;;
        *) wget -q -O /dev/null -T 10 --post-data="$2" "${FORGE_HC_BACKUP_URL}$1" || echo "[backup] could not reach Healthchecks.io" ;;
    esac
}

backup_once() {
    : >"$LOG"
    restic cat config >/dev/null 2>>"$LOG"
    status=$?
    # restic exits 10 when the repository does not exist yet.
    if [ "$status" -eq 10 ]; then
        echo "[backup] no repository yet, creating it" | tee -a "$LOG"
        restic init >>"$LOG" 2>&1 || return 1
    elif [ "$status" -ne 0 ]; then
        return 1
    fi
    restic backup /generated_envs --host forge --tag forge >>"$LOG" 2>&1 \
        && restic forget --host forge --keep-daily 7 --keep-weekly 4 --prune >>"$LOG" 2>&1
}

run() {
    if backup_once; then
        echo "[backup] done: $(grep -E '^snapshot .* saved' "$LOG" | tail -1)"
        ping_check "" "$(tail -n 5 "$LOG")"
    else
        echo "[backup] failed:" && tail -n 20 "$LOG"
        ping_check "/fail" "$(tail -n 20 "$LOG")"
        return 1
    fi
}

if [ "${1:-}" = "once" ]; then
    run
    exit $?
fi
while :; do
    run
    sleep 86400
done
