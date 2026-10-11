"""Deterministic rules for content that must never become training data.

Rules run on raw text, before scrubbing, so a payload cannot hide behind a
pseudonym. Each rule targets the dangerous form only: `rm -rf /` fires,
`rm -rf ./build` does not.
"""
from __future__ import annotations

import re

from forge.safety.types import Finding

_DETAIL_CHARS = 80
_I = re.IGNORECASE
_END = r"(?=$|[\s;&|])"

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("prompt_injection", re.compile(
        r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?(?:of\s+)?(?:the\s+|your\s+|my\s+)?"
        r"(?:previous|prior|above|earlier|preceding|system)\s+"
        r"(?:instructions?|prompts?|rules?|directions?|guidelines?)", _I)),
    ("prompt_injection", re.compile(r"<\s*/?\s*system\s*>|<\|im_(?:start|end)\|>|\[/?INST\]", _I)),
    ("prompt_injection", re.compile(r"^\s*#{2,}\s*(?:system|instructions?)\s*:", _I | re.MULTILINE)),
    ("jailbreak", re.compile(r"\bdo anything now\b|\byou are (?:now )?DAN\b", _I)),
    ("jailbreak", re.compile(r"\b(?:enable|enter|activate)\s+developer\s+mode\b", _I)),
    ("jailbreak", re.compile(
        r"\bwithout\s+any\s+(?:restrictions|filters|limitations|guidelines)\b"
        r"|\bno\s+(?:ethical|moral|safety)\s+(?:guidelines|restrictions|limits)\b", _I)),
    ("destructive_command", re.compile(
        r"\brm\s+(?:-{1,2}[\w-]+\s+)+(?:/\*?|~/?|\$HOME/?)" + _END)),
    ("destructive_command", re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")),
    ("destructive_command", re.compile(r"\bmkfs(?:\.\w+)?\s+/dev/")),
    ("destructive_command", re.compile(r"\bdd\s+[^;\n]*\bof=/dev/(?:sd|hd|nvme|xvd|disk)")),
    ("destructive_command", re.compile(r"\bchmod\s+-R\s+0?777\s+/" + _END)),
    ("remote_exec", re.compile(r"\b(?:curl|wget)\b[^\n|;]*\|\s*(?:sudo\s+)?(?:ba|z|k|da)?sh\b")),
]

# Exfiltration is a sensitive source and a network sink on the same line.
_SENSITIVE_SOURCE = re.compile(
    r"~/\.ssh/|/etc/shadow|\.aws/credentials|\bprintenv\b"
    r"|\$\{?[A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*"
)
_NETWORK_SINK = re.compile(r"\b(?:curl|wget|nc|ncat|netcat|scp|rsync)\b|/dev/tcp/")


def detect_threats(text: str) -> list[Finding]:
    """Every rule `text` breaks, with the text that broke it."""
    findings = [
        Finding(rule, match.group()[:_DETAIL_CHARS])
        for rule, pattern in _RULES
        for match in pattern.finditer(text)
    ]
    findings.extend(
        Finding("exfiltration", line.strip()[:_DETAIL_CHARS])
        for line in text.splitlines()
        if _SENSITIVE_SOURCE.search(line) and _NETWORK_SINK.search(line)
    )
    return findings
