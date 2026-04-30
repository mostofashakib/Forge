import os
from celery import Celery
from celery.schedules import crontab

celery = Celery(
    "forge",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

celery.conf.beat_schedule = {
    "cleanup-expired-sandboxes-daily": {
        "task": "backend.app.worker.tasks.cleanup_expired_sandboxes",
        "schedule": crontab(hour=2, minute=0),
    },
}
