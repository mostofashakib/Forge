"""Values passed between the safety detectors and the service."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DataSource(StrEnum):
    """Where a pipeline's data came from. It decides whether PII is scrubbed."""

    MODEL = "model"
    INTERNET = "internet"
    HUMAN = "human"

    @property
    def scrubs_pii(self) -> bool:
        return self is not DataSource.MODEL


@dataclass(frozen=True)
class Span:
    """One sensitive value found in a string, by character offsets."""

    start: int
    end: int
    kind: str
    value: str


@dataclass(frozen=True)
class Finding:
    """One reason a row is unsafe to train on."""

    rule: str
    detail: str


@dataclass(frozen=True)
class ScreenResult:
    """A scrubbed copy of a row, and whether it may enter the dataset."""

    record: object
    kept: bool
    findings: list[Finding] = field(default_factory=list)
    replacements: dict[str, int] = field(default_factory=dict)
