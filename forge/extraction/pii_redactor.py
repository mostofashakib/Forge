from __future__ import annotations
import copy
import re

from forge.extraction.schemas import CompilerInput

PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
    ("PHONE", re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]


class PIIRedactor:
    def redact(self, text: str) -> str:
        for label, pattern in PII_PATTERNS:
            text = pattern.sub(f"[REDACTED_{label}]", text)
        return text

    def redact_compiler_input(self, compiler_input: CompilerInput) -> CompilerInput:
        data = copy.deepcopy(compiler_input.model_dump())
        data = self._redact_obj(data)
        return CompilerInput.model_validate(data)

    def _redact_obj(self, obj: object) -> object:
        if isinstance(obj, dict):
            return {k: self._redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._redact_obj(item) for item in obj]
        if isinstance(obj, str):
            return self.redact(obj)
        return obj
