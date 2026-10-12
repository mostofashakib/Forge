"""Rules that keep non-permissive and proprietary code out of datasets.

Only permissively licensed code is kept. License headers carry the signal, so
proprietary notices count only inside comment lines: an email footer saying
"All rights reserved" is not a license, and BSD files say it next to a grant.
Choosing repos by license is the ingestion pipeline's job. These rules catch
files that disagree with their repo, and code a model reproduced.
"""
from __future__ import annotations

import re

from forge.safety.types import Finding

PERMISSIVE_LICENSES = frozenset(
    {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "CC0-1.0", "0BSD"}
)
_RULE = "copyright"
_DETAIL_CHARS = 80
_I = re.IGNORECASE

# An expression ends at the first character SPDX ids cannot hold: a quote,
# a comment closer, or a shell redirect.
_SPDX = re.compile(r"SPDX-License-Identifier:[ \t]*([A-Za-z0-9.+() \t-]+)")
_GPL_TERMS = re.compile(
    r"\b(?:under|terms\s+of)\s+the\s+(?:terms\s+of\s+the\s+)?GNU\s+"
    r"(?:Affero\s+|Lesser\s+|Library\s+)?General\s+Public\s+License", _I
)
_COMMENT_LINE = re.compile(r"^\s*(?:#|//|/\*|\*|--|;).*$", re.MULTILINE)
_PROPRIETARY = re.compile(
    r"proprietary\s+and\s+confidential|confidential\s+and\s+proprietary"
    r"|do\s+not\s+(?:re)?distribute|unauthorized\s+(?:copying|reproduction|distribution)\b[^\n]{0,60}prohibited", _I
)
_ALL_RIGHTS = re.compile(r"all\s+rights\s+reserved", _I)
_PERMISSIVE_GRANT = re.compile(
    r"Permission is hereby granted|Redistribution and use in source and binary forms"
    r"|Licensed under the Apache License|Permission to use, copy, modify"
    r"|free and unencumbered software"
    r"|governed by (?:a|an|the) (?:BSD|MIT|Apache|ISC)[\w-]*(?: style)? license", _I
)


def detect_copyright(text: str) -> list[Finding]:
    """Every reason `text` is not permissively licensed code."""
    findings = [
        Finding(_RULE, f"SPDX {expression.strip()}")
        for expression in _SPDX.findall(text)
        if not _spdx_allowed(expression)
    ]
    findings.extend(Finding(_RULE, match.group()[:_DETAIL_CHARS]) for match in _GPL_TERMS.finditer(text))
    granted = bool(_PERMISSIVE_GRANT.search(text)) or any(_spdx_allowed(e) for e in _SPDX.findall(text))
    for line in _COMMENT_LINE.findall(text):
        if _PROPRIETARY.search(line) or (_ALL_RIGHTS.search(line) and not granted):
            findings.append(Finding(_RULE, line.strip()[:_DETAIL_CHARS]))
    return findings


def _spdx_allowed(expression: str) -> bool:
    """An OR expression needs one allowed choice. Anything else needs every id allowed."""
    # A WITH exception narrows its base license, so only the base id counts.
    expression = re.sub(r"\s+WITH\s+[\w.+-]+", "", expression.replace("(", " ").replace(")", " "))
    tokens = re.findall(r"[A-Za-z0-9.+-]+", expression)
    ids = [token for token in tokens if token not in {"AND", "OR"}]
    if not ids:
        return False
    if "OR" in tokens:
        return any(license_id in PERMISSIVE_LICENSES for license_id in ids)
    return all(license_id in PERMISSIVE_LICENSES for license_id in ids)
