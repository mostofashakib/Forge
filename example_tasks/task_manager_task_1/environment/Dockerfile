FROM python:3.12-slim
RUN useradd -m agent
COPY fleet /opt/fleet/fleet
COPY task_manager_task_1/environment/task_manager_env.py /app/task_manager_env.py
ENV PYTHONPATH=/opt/fleet
RUN python3 /app/task_manager_env.py seed \
        --db /app/task_manager.db \
        --snapshot /app/task_manager_seed_snapshot.sql && \
    python3 /app/task_manager_env.py create-task \
        --db /app/task_manager.db \
        --task-id TASK031 \
        --title "Component design tokens" \
        --description "Define and document the design token system for the dashboard component library." \
        --project-id P004 \
        --priority MEDIUM \
        --assignee U004 \
        --actor-id U004 \
        --labels "design,frontend" && \
    python3 /app/task_manager_env.py create-task \
        --db /app/task_manager.db \
        --task-id TASK032 \
        --title "Brand refresh mockups" \
        --description "High-fidelity mockups applying brand refresh guidelines to all dashboard views." \
        --project-id P004 \
        --milestone-id M005 \
        --priority MEDIUM \
        --assignee U004 \
        --actor-id U004 \
        --labels "design" && \
    python3 /app/task_manager_env.py save-snapshot \
        --db /app/task_manager.db \
        --snapshot /app/task_manager_seed_snapshot.sql && \
    chown -R agent:agent /app
WORKDIR /app
