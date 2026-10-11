import asyncio
import threading
import pytest

from forge.runtime.error_handling import (
    CentralizedErrorHandler,
    ErrorHandler,
    ErrorRecord,
    ErrorSeverity,
    get_error_handler,
    set_default_error_handler,
)
from forge.runtime.errors import ForgeError, InvalidActionError
from forge.envgen.error_handling import GenerationErrorHandler
from forge.personas.generator import PersonaGenerator


def test_alias_and_default_instance():
    assert ErrorHandler is CentralizedErrorHandler
    handler = get_error_handler()
    assert isinstance(handler, CentralizedErrorHandler)


def test_capture_records_error_without_swallowing():
    handler = CentralizedErrorHandler(name="test_handler", default_origin="test")
    handler.clear()

    try:
        raise ValueError("Something broke")
    except ValueError as exc:
        result = handler.capture(
            exc,
            origin="custom_origin",
            code="CUSTOM_VAL_ERR",
            severity=ErrorSeverity.ERROR,
            context={"user_id": 42},
            reraise=False,
        )

    assert result is not None
    assert handler.has_errors()
    assert handler.error_count() == 1

    record = handler.last_error
    assert record is not None
    assert record.error_type == "ValueError"
    assert record.message == "Something broke"
    assert record.origin == "custom_origin"
    assert record.code == "CUSTOM_VAL_ERR"
    assert record.severity == "ERROR"
    assert record.context == {"user_id": 42}
    assert "ValueError: Something broke" in record.traceback

    # Verify serialization
    as_dict = record.to_dict()
    assert as_dict["error_type"] == "ValueError"
    assert as_dict["context"] == {"user_id": 42}


def test_capture_with_reraise():
    handler = CentralizedErrorHandler()

    with pytest.raises(RuntimeError, match="Fatal error"):
        try:
            raise RuntimeError("Fatal error")
        except RuntimeError as exc:
            handler.capture(exc, reraise=True)

    assert handler.error_count() == 1
    assert handler.last_error.error_type == "RuntimeError"


def test_capture_with_wrap_as():
    handler = CentralizedErrorHandler()

    try:
        raise KeyError("missing_key")
    except KeyError as exc:
        wrapped = handler.capture(
            exc,
            origin="environment",
            wrap_as=InvalidActionError,
            reraise=False,
        )

    assert isinstance(wrapped, InvalidActionError)
    assert wrapped.__cause__ is not None
    assert isinstance(wrapped.__cause__, KeyError)
    assert handler.error_count() == 1


def test_capture_context_manager():
    handler = CentralizedErrorHandler()

    # Case 1: reraise=False records without silent swallow
    with handler.capture_context(origin="test_ctx", severity=ErrorSeverity.WARNING, reraise=False):
        raise TypeError("wrong type")

    assert handler.error_count() == 1
    rec = handler.last_error
    assert rec.error_type == "TypeError"
    assert rec.severity == "WARNING"
    assert rec.origin == "test_ctx"

    # Case 2: reraise=True propagates
    with pytest.raises(IndexError):
        with handler.capture_context(reraise=True):
            raise IndexError("out of range")

    assert handler.error_count() == 2


def test_catch_decorator_sync():
    handler = CentralizedErrorHandler()

    @handler.catch(origin="sync_task", fallback=999, reraise=False)
    def risky_div(a, b):
        return a / b

    # Success case
    assert risky_div(10, 2) == 5.0
    assert not handler.has_errors()

    # Failure case
    result = risky_div(10, 0)
    assert result == 999
    assert handler.error_count() == 1
    assert handler.last_error.error_type == "ZeroDivisionError"


@pytest.mark.asyncio
async def test_catch_decorator_async():
    handler = CentralizedErrorHandler()

    @handler.catch(origin="async_task", fallback="default_val", reraise=False)
    async def async_fetch():
        await asyncio.sleep(0.001)
        raise ConnectionResetError("Connection lost")

    result = await async_fetch()
    assert result == "default_val"
    assert handler.error_count() == 1
    assert handler.last_error.error_type == "ConnectionResetError"


def test_handle_with_fallback():
    handler = CentralizedErrorHandler()

    def buggy_call():
        raise FileNotFoundError("missing.txt")

    fallback_val = {"status": "fallback"}
    res = handler.handle_with_fallback(
        buggy_call,
        fallback_val,
        origin="io_service",
        context={"path": "missing.txt"},
    )

    assert res == fallback_val
    assert handler.error_count() == 1
    rec = handler.last_error
    assert rec.error_type == "FileNotFoundError"
    assert rec.context == {"path": "missing.txt"}


def test_subscribe_listener():
    handler = CentralizedErrorHandler()
    captured_records: list[ErrorRecord] = []

    unsub = handler.subscribe(lambda rec: captured_records.append(rec))

    handler.capture(ValueError("first"), reraise=False)
    assert len(captured_records) == 1
    assert captured_records[0].message == "first"

    unsub()
    handler.capture(ValueError("second"), reraise=False)
    assert len(captured_records) == 1  # No new notification after unsub


def test_filtering_and_clear():
    handler = CentralizedErrorHandler()
    handler.capture(ValueError("e1"), origin="env", severity="ERROR", reraise=False)
    handler.capture(KeyError("e2"), origin="agent", severity="WARNING", reraise=False)
    handler.capture(ValueError("e3"), origin="env", severity="CRITICAL", reraise=False)

    assert len(handler.get_records(origin="env")) == 2
    assert len(handler.get_records(origin="agent")) == 1
    assert len(handler.get_records(severity="ERROR")) == 1
    assert len(handler.get_records(severity="WARNING")) == 1
    assert len(handler.get_records(error_type="ValueError")) == 2
    assert len(handler.get_records(limit=2)) == 2

    assert len(handler.to_dict()) == 3
    handler.clear()
    assert not handler.has_errors()
    assert handler.error_count() == 0


def test_generation_error_handler_integration():
    global_handler = get_error_handler()
    global_handler.clear()

    gen_handler = GenerationErrorHandler()
    gen_handler.capture(
        task_id="task_backend_1",
        agent_id="backend_builder",
        error=ValueError("Syntax error in generated backend"),
    )

    assert len(gen_handler.records) == 1
    assert global_handler.has_errors()
    last = global_handler.last_error
    assert last.origin == "environment_generation"
    assert last.context == {"task_id": "task_backend_1", "agent_id": "backend_builder"}


def test_persona_generator_records_llm_failure():
    global_handler = get_error_handler()
    global_handler.clear()

    class FailingLLM:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("API rate limit exceeded")

    gen = PersonaGenerator(llm_client=FailingLLM())
    persona = gen.generate_persona(role="Doctor", context="Hospital", seed=42)

    # Deterministic fallback still succeeds
    assert persona is not None
    assert persona.profile.role == "Doctor"

    # But the error was NOT swallowed silently!
    assert global_handler.has_errors()
    rec = global_handler.last_error
    assert rec.origin == "persona_generator"
    assert rec.error_type == "RuntimeError"
    assert "API rate limit exceeded" in rec.message


def test_concurrent_error_capture():
    handler = CentralizedErrorHandler()
    errors_to_log = 50

    def worker(worker_id: int):
        for i in range(errors_to_log):
            handler.capture(
                RuntimeError(f"Worker {worker_id} error {i}"),
                origin=f"worker_{worker_id}",
                reraise=False,
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert handler.error_count() == 4 * errors_to_log
