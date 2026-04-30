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
    # socket_connect_timeout covers the TCP handshake only; no socket_timeout so
    # Celery's read loop isn't cut short during normal operation or slow startup.
    broker_connection_timeout=4,
    broker_connection_retry=True,
    broker_connection_max_retries=2,
    broker_transport_options={
        "socket_connect_timeout": 4,
    },
)

celery.conf.beat_schedule = {
    "cleanup-expired-sandboxes-daily": {
        "task": "backend.app.worker.tasks.cleanup_expired_sandboxes",
        "schedule": crontab(hour=2, minute=0),
    },
}
