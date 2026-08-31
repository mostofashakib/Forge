#!/bin/bash
# Reference solution: find all tasks assigned to U004, branch per task on milestone presence.
# Tasks WITH milestone → reassign to U002.
# Tasks WITHOUT milestone → reassign to U003 + read existing labels + append needs-triage.

set -euo pipefail

DB=/app/task_manager.db

# Step 1: List all tasks assigned to U004
# Returns: TASK006 (M003), TASK008 (M005), TASK009 (M005), TASK031 (no milestone), TASK032 (M005)
python3 /app/task_manager_env.py list-tasks --db "$DB" --assignee U004

# Step 2: For each task, inspect it to check milestone_id
python3 /app/task_manager_env.py get-task --db "$DB" --task-id TASK006
# milestone_id: M003 → reassign to U002
python3 /app/task_manager_env.py update-task --db "$DB" --task-id TASK006 --assignee U002 --actor-id U001

python3 /app/task_manager_env.py get-task --db "$DB" --task-id TASK008
# milestone_id: M005 → reassign to U002
python3 /app/task_manager_env.py update-task --db "$DB" --task-id TASK008 --assignee U002 --actor-id U001

python3 /app/task_manager_env.py get-task --db "$DB" --task-id TASK009
# milestone_id: M005 → reassign to U002
python3 /app/task_manager_env.py update-task --db "$DB" --task-id TASK009 --assignee U002 --actor-id U001

python3 /app/task_manager_env.py get-task --db "$DB" --task-id TASK031
# milestone_id: null, labels: [design, frontend] → reassign to U003 + append needs-triage
python3 /app/task_manager_env.py update-task --db "$DB" --task-id TASK031 --assignee U003 --labels "design,frontend,needs-triage" --actor-id U001

python3 /app/task_manager_env.py get-task --db "$DB" --task-id TASK032
# milestone_id: M005 → reassign to U002
python3 /app/task_manager_env.py update-task --db "$DB" --task-id TASK032 --assignee U002 --actor-id U001

echo "TASK006,TASK008,TASK009,TASK031,TASK032"
