#!/usr/bin/env bash
# Reference solution for the channel/group name-collision task.

set -euo pipefail

DB=/app/slack.db

GROUP_RESULT="$(
  python3 /app/slack_env.py create_group \
    --db "$DB" \
    --name "security-alerts" \
    --participants "U005,U002" \
    --actor-id U002
)"
GROUP_ID="$(printf '%s' "$GROUP_RESULT" | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')"

python3 /app/slack_env.py post_message \
  --db "$DB" \
  --channel-id C005 \
  --body "Incident analysis started" \
  --actor-id U002

REPLY_RESULT="$(
  python3 /app/slack_env.py reply_to_thread \
    --db "$DB" \
    --thread-parent-id MSG004 \
    --body "On it." \
    --actor-id U002
)"
REPLY_ID="$(printf '%s' "$REPLY_RESULT" | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')"

python3 /app/slack_env.py add_reaction \
  --db "$DB" \
  --message-id "$REPLY_ID" \
  --emoji heart \
  --actor-id U002

python3 /app/slack_env.py send_group_message \
  --db "$DB" \
  --group-id "$GROUP_ID" \
  --body "Welcome @eva and @ben!" \
  --actor-id U002

python3 /app/slack_env.py change_group_name \
  --db "$DB" \
  --group-id "$GROUP_ID" \
  --new-name "Design Ops Sync" \
  --actor-id U002

printf '%s\n' "$GROUP_ID"
