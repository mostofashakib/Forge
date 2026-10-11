"""Regex detectors for structured PII.

Each pattern is tuned against what agent traces contain: timestamps, dates,
private sandbox IPs, and ids. Numeric kinds carry a checksum (Luhn, mod-97) or
a required shape (separators) so a bare number is not mistaken for a phone or
a card.
"""
from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Iterator

from forge.safety.types import Span

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# US numbers need separators so a 10-digit timestamp is not a phone.
_PHONE_US = re.compile(r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\w)")
_PHONE_E164 = re.compile(r"(?<![\w+])\+\d{10,14}(?!\d)")
_SSN = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_CARD_PREFIX = re.compile(r"^(?:4|5[1-5]|2[2-7]|3[47]|6(?:011|5))")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\.?\d)")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
_ADDRESS = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s+){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|"
    r"Place|Pl|Terrace|Ter|Parkway|Pkwy|Circle|Cir|Highway|Hwy)\b\.?"
    r"(?:,?\s+(?:Apt|Suite|Ste|Unit)\.?\s*#?\w+)?"
)


def detect_pii(text: str) -> list[Span]:
    """Every structured PII value in `text`, in no particular order."""
    spans: list[Span] = []
    for kind, pattern, accept in _DETECTORS:
        spans.extend(_matches(text, kind, pattern, accept))
    return spans


def _matches(
    text: str, kind: str, pattern: re.Pattern[str], accept: Callable[[str], bool]
) -> Iterator[Span]:
    for match in pattern.finditer(text):
        if accept(match.group()):
            yield Span(match.start(), match.end(), kind, match.group())


def _always(_: str) -> bool:
    return True


def _is_card(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 13 <= len(digits) <= 19 and bool(_CARD_PREFIX.match(digits)) and _luhn_ok(digits)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def _is_iban(value: str) -> bool:
    compact = value.replace(" ", "")
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1


def _is_public_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


_DETECTORS: list[tuple[str, re.Pattern[str], Callable[[str], bool]]] = [
    ("email", _EMAIL, _always),
    ("phone", _PHONE_US, _always),
    ("phone", _PHONE_E164, _always),
    ("ssn", _SSN, _always),
    ("card", _CARD, _is_card),
    ("iban", _IBAN, _is_iban),
    ("ip", _IPV4, _is_public_ip),
    ("ip", _IPV6, _is_public_ip),
    ("address", _ADDRESS, _always),
]
