"""Regex PII detectors: real values are found, lookalikes in traces are not."""
import pytest

from forge.safety.pii import detect_pii


def _kinds(text: str) -> list[tuple[str, str]]:
    return [(span.kind, span.value) for span in detect_pii(text)]


@pytest.mark.parametrize("text,kind,value", [
    ("mail alice.smith+ops@acme-corp.com now", "email", "alice.smith+ops@acme-corp.com"),
    ("call (415) 555-0134 today", "phone", "(415) 555-0134"),
    ("call 415-555-0134", "phone", "415-555-0134"),
    ("call +1 415.555.0134", "phone", "+1 415.555.0134"),
    ("intl +447911123456 ok", "phone", "+447911123456"),
    ("ssn 123-45-6789 on file", "ssn", "123-45-6789"),
    ("card 4111 1111 1111 1111 exp", "card", "4111 1111 1111 1111"),
    ("card 5500-0000-0000-0004", "card", "5500-0000-0000-0004"),
    ("amex 378282246310005", "card", "378282246310005"),
    ("iban GB82 WEST 1234 5698 7654 32 please", "iban", "GB82 WEST 1234 5698 7654 32"),
    ("iban DE89370400440532013000", "iban", "DE89370400440532013000"),
    ("from 8.8.8.8 again", "ip", "8.8.8.8"),
    ("v6 2606:4700:4700::1111 up", "ip", "2606:4700:4700::1111"),
    ("ship to 1600 Pennsylvania Avenue today", "address", "1600 Pennsylvania Avenue"),
    ("lives at 42 Oak St, Apt 5", "address", "42 Oak St, Apt 5"),
])
def test_each_pii_kind_is_found(text, kind, value):
    assert (kind, value) in _kinds(text)


@pytest.mark.parametrize("text", [
    "created_at 1760000000",          # unix seconds is not a phone
    "ts=1760000000123",               # unix millis is not a card
    "card 4111 1111 1111 1112",       # fails Luhn
    "host 10.0.0.1",                  # private
    "host 127.0.0.1",                 # loopback
    "host 192.168.1.5",               # private
    "due 2026-10-11",                 # date
    "at 12:30:45",                    # time
    "ticket ABC-123-4567",            # no phone shape
    "iban GB00 WEST 1234 5698 7654 32",  # fails mod-97
    "step 3 of the plan",             # number + word, no street suffix
    "ssn 000-12-3456",                # never-issued area
])
def test_trace_lookalikes_are_not_pii(text):
    assert _kinds(text) == []


def test_spans_point_at_the_value():
    text = "reach me at bob@example.org"
    (span,) = detect_pii(text)
    assert text[span.start:span.end] == "bob@example.org"


def test_empty_text_has_no_spans():
    assert detect_pii("") == []
