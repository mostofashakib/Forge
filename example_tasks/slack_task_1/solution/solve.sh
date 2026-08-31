#!/bin/bash
# Reference solution for slack_task_1 (DM/group duality, message update,
# dynamic thread replies).
#
# Trap map:
#   - Steps 1-2 use the EXISTING DM (D001) and the group containing Alice
#     (G001 "Incident Response Core"); creating chats or posting to channels
#     is forbidden. DM first, then group (order is verified).
#   - Step 3 edits Ben's OWN reply (MSG023) in the MSG002 thread; the adjacent
#     reply MSG024 belongs to the look-alike user "Ben Ortíz" and must not be
#     touched (the service would reject it anyway: author-only edits).
#   - Steps 4-5 replies depend on reading each thread's last commenter:
#     MSG004 thread -> MSG014 by Alice Nguyen; MSG010 thread -> MSG026 by
#     Ben Ortíz (the accented í must be transcribed exactly).

set -euo pipefail

DB=/app/slack.db

# Step 1: DM to Alice in the existing chat D001.
python3 /app/slack_env.py send_dm_message --db "$DB" --chat-id D001 --body "Incident review starts at noon." --actor-id U002

# Step 2: same body to the group containing Alice (G001).
python3 /app/slack_env.py send_group_message --db "$DB" --group-id G001 --body "Incident review starts at noon." --actor-id U002

# Step 3: edit Ben's own reply in the "Latency watch is normal." thread.
python3 /app/slack_env.py get_channel_messages --db "$DB" --channel-id C002
python3 /app/slack_env.py update_message --db "$DB" --message-id MSG023 --body "Graphs reviewed, latency back to baseline." --actor-id U002

# Step 4: last commenter under MSG004 is Alice Nguyen (MSG014).
python3 /app/slack_env.py get_channel_messages --db "$DB" --channel-id C003
python3 /app/slack_env.py reply_to_thread --db "$DB" --thread-parent-id MSG004 --body "Welcome Alice Nguyen" --actor-id U002

# Step 5: last commenter under MSG010 is Ben Ortíz (MSG026, the decoy user).
python3 /app/slack_env.py get_channel_messages --db "$DB" --channel-id C001
python3 /app/slack_env.py reply_to_thread --db "$DB" --thread-parent-id MSG010 --body "Welcome Ben Ortíz" --actor-id U002

# Step 6: final answer.
echo "Alice Nguyen;Ben Ortíz"
