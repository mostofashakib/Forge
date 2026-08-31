FROM python:3.12-slim
RUN useradd -m agent
COPY fleet /opt/fleet/fleet
COPY slack_task_1/environment/slack_env.py /app/slack_env.py
ENV PYTHONPATH=/opt/fleet
RUN python3 /app/slack_env.py seed \
    --db /app/slack.db \
    --snapshot /app/slack_seed_snapshot.sql \
    && chown -R agent:agent /app
WORKDIR /app
