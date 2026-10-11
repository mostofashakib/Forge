import asyncio
import io
import logging
import threading
import pytest

from forge import DevLogger, DeveloperLogger
from forge.dev_logger import DevLogRecord
from forge.runtime.error_handling import get_error_handler


def test_dev_logger_singleton_and_alias():
    assert DeveloperLogger is DevLogger
    l1 = DevLogger.get("test_singleton")
    l2 = DevLogger.get("test_singleton")
    assert l1 is l2
    assert l1.name == "test_singleton"


def test_dev_logger_structured_context():
    stream = io.StringIO()
    logger = DevLogger("test_context", level=logging.INFO, stream=stream)
    logger.clear()

    logger.info("Task dispatched", task_id="t_123", worker_id=4)

    assert logger.has_logged("Task dispatched")
    rec = logger.last_record
    assert rec is not None
    assert rec.message == "Task dispatched"
    assert rec.context["task_id"] == "t_123"
    assert rec.context["worker_id"] == 4
    assert "task_id='t_123'" in stream.getvalue()


def test_dev_logger_bind():
    logger = DevLogger("test_bind")
    child = logger.bind(env_name="crm_v1", region="us-east-1")
    child.clear()

    child.info("Agent started", step=0)

    rec = child.last_record
    assert rec is not None
    assert rec.context["env_name"] == "crm_v1"
    assert rec.context["region"] == "us-east-1"
    assert rec.context["step"] == 0


def test_dev_logger_context_manager():
    logger = DevLogger("test_ctx_mgr")
    logger.clear()

    with logger.context(episode_id="ep_001"):
        logger.info("Inside block 1")
        with logger.context(step=5):
            logger.info("Inside nested block")
        logger.info("Inside block 2")

    logger.info("Outside block")

    records = logger.records
    assert len(records) == 4
    assert records[0].context == {"episode_id": "ep_001"}
    assert records[1].context == {"episode_id": "ep_001", "step": 5}
    assert records[2].context == {"episode_id": "ep_001"}
    assert records[3].context == {}


def test_dev_logger_sensitive_redaction():
    logger = DevLogger("test_redact")
    logger.clear()

    secret_url = "https://example.com/api?token=secret123&other=val"
    logger.warning("Connecting with %s and Bearer super_secret_token", secret_url)

    rec = logger.last_record
    assert rec is not None
    assert "?[REDACTED]" in rec.message
    assert "secret123" not in rec.message
    assert "Bearer [REDACTED]" in rec.message
    assert "super_secret_token" not in rec.message


def test_dev_logger_timer():
    logger = DevLogger("test_timer")
    logger.clear()

    with logger.timer("expensive_calc", env="prod"):
        import time
        time.sleep(0.01)

    assert len(logger.records) == 2
    assert logger.records[0].message == "Started: expensive_calc"
    assert logger.records[1].message.startswith("Finished: expensive_calc")
    assert logger.records[1].duration_ms is not None
    assert logger.records[1].duration_ms >= 5.0
    assert logger.records[1].context == {"env": "prod"}


def test_dev_logger_sync_timed_decorator():
    logger = DevLogger("test_sync_timed")
    logger.clear()

    @logger.timed(label="sync_task", component="compiler")
    def run_sync(x: int) -> int:
        return x * 2

    res = run_sync(5)
    assert res == 10
    assert len(logger.records) == 2
    assert "Started: sync_task" in logger.records[0].message
    assert "Finished: sync_task" in logger.records[1].message
    assert logger.records[1].context == {"component": "compiler"}


@pytest.mark.asyncio
async def test_dev_logger_async_timed_decorator():
    logger = DevLogger("test_async_timed")
    logger.clear()

    @logger.timed(label="async_op")
    async def run_async():
        await asyncio.sleep(0.01)
        return "done"

    res = await run_async()
    assert res == "done"
    assert len(logger.records) == 2
    assert "Started: async_op" in logger.records[0].message
    assert "Finished: async_op" in logger.records[1].message


def test_dev_logger_exception_centralized_integration():
    central_handler = get_error_handler()
    central_handler.clear()

    logger = DevLogger("test_exc")
    logger.clear()

    try:
        raise ValueError("Invalid configuration for container")
    except ValueError as exc:
        logger.exception("Failed to initialize container", exc_info=exc, container_id="c_1")

    assert logger.has_logged("Failed to initialize container", level="ERROR")
    rec = logger.last_record
    assert rec.exc_info is not None
    assert "ValueError: Invalid configuration for container" in rec.exc_info

    # Verify central handler received error
    assert central_handler.has_errors()
    last_err = central_handler.last_error
    assert last_err.origin == "test_exc"
    assert last_err.message == "Invalid configuration for container"


def test_dev_logger_helpers_and_clear():
    logger = DevLogger("test_helpers")
    logger.clear()

    logger.debug("Debug msg")
    logger.info("Info msg")
    logger.error("Error msg")

    assert logger.has_logged("Debug msg", level="DEBUG")
    assert logger.has_logged("Info msg", level="INFO")
    assert not logger.has_logged("Info msg", level="ERROR")
    assert len(logger.records) == 3

    as_dict = logger.records[0].to_dict()
    assert as_dict["level"] == "DEBUG"
    assert as_dict["message"] == "Debug msg"

    logger.clear()
    assert len(logger.records) == 0
    assert logger.last_record is None


def test_dev_logger_thread_safety():
    logger = DevLogger("test_concurrency")
    logger.clear()

    def worker(idx: int):
        for i in range(25):
            logger.info(f"Worker {idx} message {i}", worker=idx, iter=i)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(logger.records) == 100
