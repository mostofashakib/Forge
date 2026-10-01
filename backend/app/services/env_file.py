"""Read and rewrite named keys in backend/.env, leaving every other line alone.

The settings page saves through `update_env_file`. Values are restricted to
a safe character set, so a value can never add a line or change how the
file parses. The new file is written beside the old one and swapped in.
"""
from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from dotenv import dotenv_values

BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

_KEY = re.compile(r"[A-Z_][A-Z0-9_]*")
_VALUE = re.compile(r"[A-Za-z0-9._:/@+-]*")
_ASSIGNMENT = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=)")


def read_env_values(path: Path, keys: Iterable[str]) -> dict[str, str]:
    """The asked keys that the file sets, as the file sets them."""
    if not path.exists():
        return {}
    values = dotenv_values(path)
    return {key: values[key] or "" for key in keys if key in values}


def update_env_file(path: Path, updates: Mapping[str, str]) -> None:
    for key, value in updates.items():
        if not _KEY.fullmatch(key):
            raise ValueError(f"{key!r} is not a valid setting name")
        if not _VALUE.fullmatch(value):
            raise ValueError(f"{key} may only contain letters, digits and . _ : / @ + -")
    lines = path.read_text().splitlines(keepends=True) if path.exists() else []
    remaining = dict(updates)
    rewritten = []
    for line in lines:
        match = _ASSIGNMENT.match(line)
        if match and match.group(2) in updates:
            key = match.group(2)
            ending = "\n" if line.endswith("\n") else ""
            rewritten.append(f"{match.group(1)}{key}{match.group(3)}{updates[key]}{ending}")
            remaining.pop(key, None)
        else:
            rewritten.append(line)
    if remaining and rewritten and not rewritten[-1].endswith("\n"):
        rewritten[-1] += "\n"
    rewritten += [f"{key}={value}\n" for key, value in remaining.items()]
    _atomic_write(path, "".join(rewritten))


def _atomic_write(path: Path, text: str) -> None:
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.chmod(temp, mode)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def environ_with_saved(keys: Iterable[str]) -> dict[str, str]:
    """The process environment, with `keys` as backend/.env sets them now.

    The settings page writes the file while the API and workers run, so the
    file is read at call time and wins over the value loaded at startup.
    """
    return {**os.environ, **read_env_values(BACKEND_ENV_FILE, keys)}
