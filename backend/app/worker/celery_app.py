import logging
import os
import threading

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready

log = logging.getLogger(__name__)

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


@worker_ready.connect
def _prewarm_base_images_on_boot(**_kwargs) -> None:
    """Pull Forge's standard base images once when the worker comes up.

    Runs in a daemon thread so it doesn't block the worker from accepting
    jobs. Subsequent build_sandbox_task invocations find the canonical
    python base in the local cache, so Docker Hub is never on the hot
    path of a user-triggered build — making the system immune to
    transient Hub EOF outages.

    Disabled when FORGE_DISABLE_PREWARM=1 (used in tests / local dev where
    Docker isn't available).
    """
    if os.environ.get("FORGE_DISABLE_PREWARM") == "1":
        log.info("[prewarm] disabled by FORGE_DISABLE_PREWARM=1")
        return

    def _run():
        try:
            from forge.envgen.container import prewarm_standard_base_images
            log.info("[prewarm] starting base-image pre-warm")
            results = prewarm_standard_base_images()
            log.info("[prewarm] complete: %s", results)
        except Exception as exc:  # noqa: BLE001 — never crash the worker on prewarm
            log.warning("[prewarm] aborted: %s", exc)

    threading.Thread(target=_run, name="forge-prewarm", daemon=True).start()
