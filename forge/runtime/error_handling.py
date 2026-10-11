from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import functools
import inspect
import logging
import threading
import traceback
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar
import uuid

if TYPE_CHECKING:
    from forge.runtime.errors import ForgeError

logger = logging.getLogger("forge.errors")

T = TypeVar("T")


class ErrorSeverity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ErrorRecord:
    """Immutable audit entry for a captured error."""

    error_id: str
    timestamp: datetime
    origin: str
    error_type: str
    message: str
    code: str
    traceback: str
    severity: str
    context: dict[str, Any] = field(default_factory=dict)
    cause: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "origin": self.origin,
            "error_type": self.error_type,
            "message": self.message,
            "code": self.code,
            "traceback": self.traceback,
            "severity": self.severity,
            "context": dict(self.context),
            "cause": self.cause,
        }


class CentralizedErrorHandler:
    """Centralized error handling and tracking service.

    Ensures exceptions across environments, agents, verifiers, and runtime
    services are never silently swallowed. Every error is captured, structured,
    logged with full stack trace, and broadcast to subscribers.
    """

    def __init__(
        self,
        *,
        name: str = "central",
        default_origin: str = "runtime",
        app_logger: logging.Logger | None = None,
    ) -> None:
        self.name = name
        self.default_origin = default_origin
        self.logger = app_logger or logger
        self._records: list[ErrorRecord] = []
        self._listeners: list[Callable[[ErrorRecord], None]] = []
        self._lock = threading.Lock()

    def capture(
        self,
        error: BaseException,
        *,
        origin: str | None = None,
        code: str | None = None,
        severity: str | ErrorSeverity = ErrorSeverity.ERROR,
        context: dict[str, Any] | None = None,
        reraise: bool = False,
        wrap_as: type[ForgeError] | None = None,
    ) -> ForgeError | BaseException:
        """Capture, log, and record an error with full context."""
        sev_str = severity.value if isinstance(severity, ErrorSeverity) else str(severity).upper()
        orig = origin or getattr(error, "origin", None) or self.default_origin
        err_code = code or getattr(error, "code", None) or type(error).__name__
        ctx = dict(context or {})

        tb_str = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        cause_repr = repr(error.__cause__) if error.__cause__ is not None else None

        record = ErrorRecord(
            error_id=f"err_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc),
            origin=orig,
            error_type=type(error).__name__,
            message=str(error),
            code=err_code,
            traceback=tb_str,
            severity=sev_str,
            context=ctx,
            cause=cause_repr,
        )

        with self._lock:
            self._records.append(record)

        log_level = getattr(logging, sev_str, logging.ERROR)
        self.logger.log(
            log_level,
            "[%s] %s in %s: %s (code=%s, id=%s)",
            sev_str,
            type(error).__name__,
            orig,
            error,
            err_code,
            record.error_id,
            exc_info=error,
        )

        for listener in list(self._listeners):
            try:
                listener(record)
            except Exception as listener_exc:
                self.logger.warning("Error listener failed: %s", listener_exc)

        result_err: ForgeError | BaseException = error
        if wrap_as is not None:
            from forge.runtime.errors import ForgeError
            if not issubclass(wrap_as, ForgeError):
                raise TypeError(f"wrap_as must be a subclass of ForgeError, got {wrap_as}")
            try:
                result_err = wrap_as(
                    str(error),
                    code=err_code,
                    origin=orig,
                    cause=error,
                )
            except TypeError:
                try:
                    result_err = wrap_as(str(error), code=err_code)
                except TypeError:
                    result_err = wrap_as(str(error))
            if getattr(result_err, "__cause__", None) is None:
                result_err.__cause__ = error

        if reraise:
            raise result_err from error

        return result_err

    @contextmanager
    def capture_context(
        self,
        *,
        origin: str | None = None,
        code: str | None = None,
        severity: str | ErrorSeverity = ErrorSeverity.ERROR,
        context: dict[str, Any] | None = None,
        reraise: bool = True,
        wrap_as: type[ForgeError] | None = None,
    ) -> Iterator[None]:
        """Context manager to intercept and record errors instead of swallowing them."""
        try:
            yield
        except Exception as exc:
            self.capture(
                exc,
                origin=origin,
                code=code,
                severity=severity,
                context=context,
                reraise=reraise,
                wrap_as=wrap_as,
            )

    def catch(
        self,
        *,
        origin: str | None = None,
        code: str | None = None,
        severity: str | ErrorSeverity = ErrorSeverity.ERROR,
        context: dict[str, Any] | None = None,
        reraise: bool = True,
        fallback: Any = None,
        wrap_as: type[ForgeError] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for sync or async functions that ensures errors are centralized."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as exc:
                        self.capture(
                            exc,
                            origin=origin,
                            code=code,
                            severity=severity,
                            context=context,
                            reraise=reraise,
                            wrap_as=wrap_as,
                        )
                        return fallback
                return async_wrapper
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        return fn(*args, **kwargs)
                    except Exception as exc:
                        self.capture(
                            exc,
                            origin=origin,
                            code=code,
                            severity=severity,
                            context=context,
                            reraise=reraise,
                            wrap_as=wrap_as,
                        )
                        return fallback
                return sync_wrapper
        return decorator

    def handle_with_fallback(
        self,
        func: Callable[[], T],
        fallback: T,
        *,
        origin: str | None = None,
        severity: str | ErrorSeverity = ErrorSeverity.WARNING,
        context: dict[str, Any] | None = None,
    ) -> T:
        """Execute a callable with fallback, recording any failure instead of swallowing it."""
        try:
            return func()
        except Exception as exc:
            self.capture(
                exc,
                origin=origin,
                severity=severity,
                context=context,
                reraise=False,
            )
            return fallback

    def subscribe(self, listener: Callable[[ErrorRecord], None]) -> Callable[[], None]:
        """Register a callback for all captured errors. Returns an unsubscribe function."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    @property
    def records(self) -> tuple[ErrorRecord, ...]:
        """All recorded errors as an immutable tuple."""
        with self._lock:
            return tuple(self._records)

    @property
    def last_error(self) -> ErrorRecord | None:
        """The most recently captured error record, if any."""
        with self._lock:
            return self._records[-1] if self._records else None

    def get_records(
        self,
        *,
        origin: str | None = None,
        severity: str | None = None,
        error_type: str | None = None,
        limit: int | None = None,
    ) -> list[ErrorRecord]:
        """Query captured error records with optional filtering."""
        with self._lock:
            items = list(self._records)
        if origin is not None:
            items = [r for r in items if r.origin == origin]
        if severity is not None:
            sev_norm = severity.upper()
            items = [r for r in items if r.severity == sev_norm]
        if error_type is not None:
            items = [r for r in items if r.error_type == error_type]
        if limit is not None:
            items = items[-limit:]
        return items

    def has_errors(self) -> bool:
        """Check whether any errors have been recorded."""
        with self._lock:
            return len(self._records) > 0

    def error_count(self) -> int:
        """Total number of recorded errors."""
        with self._lock:
            return len(self._records)

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize all recorded errors to dictionaries."""
        return [record.to_dict() for record in self.records]

    def clear(self) -> None:
        """Clear all recorded errors."""
        with self._lock:
            self._records.clear()


# Alias for convenience
ErrorHandler = CentralizedErrorHandler

_DEFAULT_ERROR_HANDLER: CentralizedErrorHandler | None = None
_DEFAULT_LOCK = threading.Lock()


def get_error_handler() -> CentralizedErrorHandler:
    """Retrieve the global default CentralizedErrorHandler instance."""
    global _DEFAULT_ERROR_HANDLER
    if _DEFAULT_ERROR_HANDLER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_ERROR_HANDLER is None:
                _DEFAULT_ERROR_HANDLER = CentralizedErrorHandler()
    return _DEFAULT_ERROR_HANDLER


def set_default_error_handler(handler: CentralizedErrorHandler) -> None:
    """Configure the global default CentralizedErrorHandler instance."""
    global _DEFAULT_ERROR_HANDLER
    with _DEFAULT_LOCK:
        _DEFAULT_ERROR_HANDLER = handler
