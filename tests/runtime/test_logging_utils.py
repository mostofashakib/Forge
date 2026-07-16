import logging

from forge.logging_utils import SensitiveLogFilter, redact_sensitive_text


SIGNED_URL = (
    "https://cdn.example.test/blob/data?Expires=123&Signature=secret-value"
    "&Key-Pair-Id=key-id"
)


def test_redacts_signed_url_query_string():
    result = redact_sensitive_text(f"download failed: {SIGNED_URL}")

    assert result == "download failed: https://cdn.example.test/blob/data?[REDACTED]"
    assert "secret-value" not in result
    assert "key-id" not in result


def test_redacts_bearer_credentials_outside_urls():
    result = redact_sensitive_text("Authorization: Bearer abc.def_123")

    assert result == "Authorization: Bearer [REDACTED]"


def test_log_filter_sanitizes_format_arguments():
    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="HTTP Request: GET %s",
        args=(SIGNED_URL,),
        exc_info=None,
    )

    assert SensitiveLogFilter().filter(record) is True
    assert "Signature" not in record.getMessage()
    assert "?[REDACTED]" in record.getMessage()


def test_log_filter_preserves_numeric_format_arguments():
    record = logging.LogRecord(
        name="forge.envgen._image_pull_http",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="layer %d (%.1f MiB)",
        args=(2, 10.5),
        exc_info=None,
    )

    SensitiveLogFilter().filter(record)
    assert record.getMessage() == "layer 2 (10.5 MiB)"
