#!/bin/bash
# Deterministic reference solution. Scores exactly 1.000.
#
# Drives the same `tasks` CLI the agent has, so this is a route through the
# workspace rather than a privileged shortcut: it reads the set, branches each
# task on the milestone field, and reports what it did. Nothing here touches the
# database, the seed, or the grader.
set -euo pipefail

# The five tasks Jordan Kim holds. Found rather than assumed -- the same call an
# agent would make, and the one the roster milestone listens for.
tasks list_tasks --assignee U004 >/dev/null

# Milestone tasks go to Morgan Patel. Each is read before it is moved, so the
# branch is taken on the record rather than on this script's memory of it.
for task in TASK006 TASK008 TASK009 TASK032; do
  tasks get_task --task-id "$task" >/dev/null
  tasks update_task --task-id "$task" --assignee U002 >/dev/null
done

# TASK031 has no milestone, so it goes to Riley Stone. `labels` replaces the
# list rather than merging into it, which is why the existing labels are read
# first and passed back alongside the new one.
tasks get_task --task-id TASK031 >/dev/null
tasks update_task \
  --task-id TASK031 \
  --assignee U003 \
  --labels "design,frontend,needs-triage" >/dev/null

# The graded answer. It lives in the world, not in this script's stdout: the
# verifier reads the workspace, never the agent's own logs.
tasks submit_handover_report \
  --task-ids "TASK006,TASK008,TASK009,TASK031,TASK032" \
  --summary "Reassigned Jordan Kim's five tasks: four milestone tasks to Morgan Patel, TASK031 to Riley Stone with needs-triage appended." >/dev/null

echo "TASK006,TASK008,TASK009,TASK031,TASK032"
