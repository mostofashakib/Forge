from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit, urlunsplit


_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SENSITIVE_PARAM_RE = re.compile(
    r"(?i)(Signature|X-Amz-Signature|Key-Pair-Id|token|access_token)=([^&\s]+)"
)


def redact_sensitive_text(value: object) -> str:
    """Remove credentials and URL query strings before text reaches logs or errors."""

    text = str(value)

    def redact_url(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        trailing = ""
        while raw_url and raw_url[-1] in ".,;:)]}":
            trailing = raw_url[-1] + trailing
            raw_url = raw_url[:-1]
        try:
            parts = urlsplit(raw_url)
        except ValueError:
            return "[REDACTED_URL]" + trailing
        if not parts.query:
            return raw_url + trailing
        safe_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "[REDACTED]", ""))
        return safe_url + trailing

    text = _URL_RE.sub(redact_url, text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    return _SENSITIVE_PARAM_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)


class SensitiveLogFilter(logging.Filter):
    """Sanitize formatted log arguments without changing call-site behavior."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive_text(record.msg)
        if isinstance(record.args, dict):
            record.args = {
                key: self._redact_argument(value) for key, value in record.args.items()
            }
        elif isinstance(record.args, tuple):
            record.args = tuple(self._redact_argument(value) for value in record.args)
        return True

    @staticmethod
    def _redact_argument(value: object) -> object:
        # Preserve numeric types so `%d`/`%f` logging placeholders remain valid.
        if isinstance(value, str) or isinstance(value, BaseException):
            return redact_sensitive_text(value)
        return value


def install_sensitive_log_filter(*logger_names: str) -> None:
    """Install one redaction filter on each named third-party logger."""

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        if not any(isinstance(item, SensitiveLogFilter) for item in logger.filters):
            logger.addFilter(SensitiveLogFilter())
