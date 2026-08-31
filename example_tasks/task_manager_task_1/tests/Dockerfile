FROM python:3.12-slim
RUN mkdir -p /logs/agent /logs/verifier /tests
RUN pip install --no-cache-dir harbor-rewardkit==0.1.4
COPY fleet /opt/fleet/fleet
COPY task_manager_task_1/tests/test.sh /tests/test.sh
COPY task_manager_task_1/tests/check.py /tests/check.py
ENV PYTHONPATH=/opt/fleet
RUN chmod +x /tests/test.sh
WORKDIR /app
