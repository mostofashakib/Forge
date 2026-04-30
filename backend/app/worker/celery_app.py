import os
from celery import Celery
from celery.schedules import crontab

celery = Celery(
    "forge",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["backend.app.worker.tasks"],
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Fail fast on initial broker connection so HTTP callers get a quick 503.
    broker_connection_timeout=4,
    broker_connection_retry=True,
    broker_connection_max_retries=2,
    broker_transport_options={
        "socket_connect_timeout": 4,
    },
    # Keep Redis backend connections alive to avoid "Connection lost" log spam
    # during long-running tasks (build_sandbox_task can take several minutes).
    result_backend_transport_options={
        "socket_keepalive": True,
        "retry_on_timeout": True,
    },
    redis_socket_keepalive=True,
    redis_retry_on_timeout=True,
)

celery.conf.beat_schedule = {
    "cleanup-expired-sandboxes-daily": {
        "task": "backend.app.worker.tasks.cleanup_expired_sandboxes",
        "schedule": crontab(hour=2, minute=0),
    },
}
