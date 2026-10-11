from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
import inspect
import logging
import sys
import threading
import time
import traceback
from typing import Any, Callable, Iterator, TextIO

from forge.logging_utils import SensitiveLogFilter, redact_sensitive_text

_context_var: ContextVar[dict[str, Any]] = ContextVar("dev_logger_context", default={})


@dataclass(frozen=True)
class DevLogRecord:
    """Structured in-memory log entry recorded by DevLogger."""

    name: str
    level: str
    message: str
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    exc_info: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": dict(self.context),
            "duration_ms": self.duration_ms,
            "exc_info": self.exc_info,
        }


class DevLogger:
    """Developer-centric structured logger for Forge.

    Provides ergonomic contextual logging, automatic credential redaction,
    execution timers, and integration with Forge's centralized error tracking.

    Usage:
        logger = DevLogger.get(__name__)
        logger.info("Environment starting", env_name="my_env", port=8000)

        with logger.timer("docker_build"):
            build_container()
    """

    _instances: dict[str, DevLogger] = {}
    _registry_lock = threading.Lock()

    def __init__(
        self,
        name: str = "forge",
        *,
        level: int | str = logging.INFO,
        stream: TextIO | None = None,
        context: dict[str, Any] | None = None,
        max_retained_records: int = 1000,
    ) -> None:
        self.name = name
        self.max_retained_records = max_retained_records
        self._bound_context = dict(context or {})
        self._records: list[DevLogRecord] = []
        self._lock = threading.Lock()

        self._logger = logging.getLogger(name)
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(level)

        if not any(isinstance(f, SensitiveLogFilter) for f in self._logger.filters):
            self._logger.addFilter(SensitiveLogFilter())

        if not self._logger.handlers:
            handler = logging.StreamHandler(stream or sys.stderr)
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            handler.addFilter(SensitiveLogFilter())
            self._logger.addHandler(handler)

    @classmethod
    def get(cls, name: str = "forge", **kwargs: Any) -> DevLogger:
        """Retrieve or create a cached DevLogger instance for the given module name."""
        with cls._registry_lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name=name, **kwargs)
            return cls._instances[name]

    def bind(self, **kwargs: Any) -> DevLogger:
        """Create a child logger with additional persistent context fields."""
        merged_context = {**self._bound_context, **kwargs}
        child = DevLogger(
            name=self.name,
            level=self._logger.level,
            context=merged_context,
            max_retained_records=self.max_retained_records,
        )
        child._logger = self._logger
        return child

    @contextmanager
    def context(self, **kwargs: Any) -> Iterator[None]:
        """Scoped context manager that attaches fields to all log messages in block."""
        current = _context_var.get()
        token = _context_var.set({**current, **kwargs})
        try:
            yield
        finally:
            _context_var.reset(token)

    def _log_internal(
        self,
        level_num: int,
        level_name: str,
        msg: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        duration_ms: float | None = None,
    ) -> None:
        exc_info = kwargs.pop("exc_info", None)
        stack_info = kwargs.pop("stack_info", False)
        extra = kwargs.pop("extra", None) or {}

        # Merge bound context, ambient contextvar, and call-site kwargs
        merged_context: dict[str, Any] = {
            **self._bound_context,
            **_context_var.get(),
            **kwargs,
        }

        # Format message if args provided
        formatted_msg = msg % args if args else msg
        redacted_msg = redact_sensitive_text(formatted_msg)

        if merged_context:
            context_str = " ".join(f"{k}={v!r}" for k, v in merged_context.items())
            log_output = f"{redacted_msg} [{context_str}]"
        else:
            log_output = redacted_msg

        exc_text: str | None = None
        if exc_info:
            if isinstance(exc_info, BaseException):
                exc_text = "".join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
            elif isinstance(exc_info, tuple):
                exc_text = "".join(traceback.format_exception(*exc_info))
            else:
                exc_text = traceback.format_exc()

        record = DevLogRecord(
            name=self.name,
            level=level_name,
            message=redacted_msg,
            timestamp=datetime.now(timezone.utc),
            context=merged_context,
            duration_ms=duration_ms,
            exc_info=exc_text,
        )

        with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_retained_records:
                self._records = self._records[-self.max_retained_records:]

        self._logger.log(
            level_num,
            log_output,
            exc_info=exc_info,
            stack_info=stack_info,
            extra=extra,
        )

        # Notify centralized error handler on ERROR/CRITICAL
        if level_num >= logging.ERROR and exc_info:
            self._notify_error_handler(record, exc_info)

    def _notify_error_handler(self, record: DevLogRecord, exc_info: Any) -> None:
        try:
            from forge.runtime.error_handling import get_error_handler
            err = exc_info if isinstance(exc_info, BaseException) else RuntimeError(record.message)
            get_error_handler().capture(
                err,
                origin=record.name,
                code=f"DEV_LOG_{record.level}",
                severity=record.level,
                context=record.context,
                reraise=False,
            )
        except Exception:
            pass

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_internal(logging.DEBUG, "DEBUG", msg, args, kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_internal(logging.INFO, "INFO", msg, args, kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_internal(logging.WARNING, "WARNING", msg, args, kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_internal(logging.ERROR, "ERROR", msg, args, kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_internal(logging.CRITICAL, "CRITICAL", msg, args, kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs["exc_info"] = kwargs.get("exc_info", True)
        self._log_internal(logging.ERROR, "ERROR", msg, args, kwargs)

    @contextmanager
    def timer(
        self,
        label: str,
        *,
        level: int | str = logging.INFO,
        **context: Any,
    ) -> Iterator[None]:
        """Context manager that measures duration and logs start/finish."""
        level_num = getattr(logging, level.upper(), logging.INFO) if isinstance(level, str) else level
        level_name = logging.getLevelName(level_num)

        start_time = time.perf_counter()
        self._log_internal(level_num, level_name, f"Started: {label}", (), context)
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._log_internal(
                level_num,
                level_name,
                f"Finished: {label} ({elapsed_ms:.2f}ms)",
                (),
                context,
                duration_ms=round(elapsed_ms, 2),
            )

    def timed(
        self,
        label: str | None = None,
        *,
        level: int | str = logging.INFO,
        **context: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for sync and async functions to log execution duration."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            op_label = label or fn.__qualname__

            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.timer(op_label, level=level, **context):
                        return await fn(*args, **kwargs)
                return async_wrapper
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.timer(op_label, level=level, **context):
                        return fn(*args, **kwargs)
                return sync_wrapper

        return decorator

    @property
    def records(self) -> tuple[DevLogRecord, ...]:
        """Immutable tuple of in-memory log records."""
        with self._lock:
            return tuple(self._records)

    @property
    def last_record(self) -> DevLogRecord | None:
        """Most recent log entry, if any."""
        with self._lock:
            return self._records[-1] if self._records else None

    def has_logged(self, text: str, level: str | None = None) -> bool:
        """Check whether any recorded message contains the given substring."""
        with self._lock:
            records = list(self._records)
        target_level = level.upper() if level else None
        for record in records:
            if target_level and record.level != target_level:
                continue
            if text in record.message:
                return True
        return False

    def clear(self) -> None:
        """Clear all in-memory records."""
        with self._lock:
            self._records.clear()


# Alias for convenience
DeveloperLogger = DevLogger
