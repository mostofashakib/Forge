"""Rewrite a generated app's Dockerfile so every build is pinned and reproducible.

The LLM that writes a Dockerfile picks its own base image, port and install
steps. These normalisers force the one pre-warmed base, the one published
port, and the one hashed install of the runtime lock.
"""
from __future__ import annotations

import re
from pathlib import Path

from forge.envgen.container_specs import FORGE_APP_PORT
from forge.envgen.docker_images import FORGE_PYTHON_BASE
from forge.envgen.runtime_lock import RUNTIME_LOCK

# The only install step a Forge Dockerfile may contain. `--require-hashes`
# makes pip refuse anything the lock does not pin to an exact artifact.
_LOCKED_INSTALL = "RUN pip install --no-cache-dir --require-hashes -r requirements.txt"

DEFAULT_DOCKERFILE = f"""\
FROM {FORGE_PYTHON_BASE}
WORKDIR /app
COPY requirements.txt .
{_LOCKED_INSTALL}
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""


# Matches FROM lines whose base is a python:* tag (any tag, with or without digest),
# optionally followed by `AS <stage>`. Non-python bases are left untouched.
_PYTHON_FROM_RE = re.compile(
    r"^(\s*FROM\s+)python:[^\s]+(\s+AS\s+\S+)?\s*$",
    re.IGNORECASE,
)


def write_locked_requirements(app_dir: Path) -> bool:
    """Make requirements.txt the runtime lock, whatever the LLM listed.

    An unpinned requirements file resolves against whatever PyPI serves on
    build day. The correctness gate rejects generated code that imports a
    package the lock lacks, so replacing the file never drops a real need.
    Returns True if the file was created or changed.
    """
    lock = RUNTIME_LOCK.read_text()
    req_file = app_dir / "requirements.txt"
    if req_file.exists() and req_file.read_text() == lock:
        return False
    req_file.write_text(lock)
    return True


# `pip install` in any spelling (pip, pip3, python -m pip).
_PIP_INSTALL_RE = re.compile(r"\bpip3?\s+install\b")
# System package managers resolve against live distro mirrors.
_SYSTEM_INSTALL_RE = re.compile(r"\b(apt-get|apt|apk)\s+(update|install|add)\b")


def _dockerfile_instructions(text: str) -> list[str]:
    """Split a Dockerfile into instructions, keeping `\\` continuations whole."""
    instructions: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        current.append(line)
        if not line.rstrip().endswith("\\"):
            instructions.append("\n".join(current))
            current = []
    if current:
        instructions.append("\n".join(current))
    return instructions


def normalise_dockerfile_install(dockerfile: Path) -> bool:
    """Reduce every install step to the single hashed install of the lock.

    The first RUN that calls pip becomes `_LOCKED_INSTALL`. Later pip RUNs
    and every system package install are dropped: the lock is the whole
    dependency set, and nothing in it needs a system package.
    Returns True if the file was modified.
    """
    original = dockerfile.read_text()
    kept: list[str] = []
    installed = False
    for instruction in _dockerfile_instructions(original):
        is_run = instruction.lstrip().upper().startswith("RUN ")
        if is_run and _PIP_INSTALL_RE.search(instruction):
            if not installed:
                kept.append(_LOCKED_INSTALL)
                installed = True
            continue
        if is_run and _SYSTEM_INSTALL_RE.search(instruction):
            continue
        kept.append(instruction)
    text = "\n".join(kept) + "\n"
    if text == original:
        return False
    dockerfile.write_text(text)
    return True

# Match a whole EXPOSE line (case-insensitive). Uses `[ \t]` instead of `\s`
# for the trailing whitespace so the match can't slurp across the newline
# into the next directive.
_EXPOSE_RE = re.compile(r"^[ \t]*EXPOSE[ \t]+\d+[ \t]*$", re.MULTILINE | re.IGNORECASE)

# Match `--port` followed (after any non-word run — spaces, commas, equals,
# quotes) by a port number. Catches every shell, JSON-array, and
# equals-form CMD the LLM might emit:
#   CMD uvicorn --port 5000
#   CMD ["uvicorn", "--port", "5000"]
#   CMD uvicorn --port=5000
_PORT_FLAG_RE = re.compile(r"(--port)(\W+?)(\d+)")


def normalise_dockerfile_port(dockerfile: Path) -> bool:
    """Force every EXPOSE and `--port` in the Dockerfile to FORGE_APP_PORT.

    The LLM that writes the Dockerfile picks ports probabilistically — sometimes
    8000, sometimes 5000, sometimes 8080. Forge always publishes 8000/tcp on the
    container side, so anything else means a dead port mapping. We rewrite the
    file to keep the two ends consistent regardless of what the LLM chose.

    Returns True if any change was made.
    """
    original = dockerfile.read_text()
    text = original

    # Rewrite any EXPOSE line to use the canonical port.
    text, expose_subs = _EXPOSE_RE.subn(f"EXPOSE {FORGE_APP_PORT}", text)
    # Add EXPOSE if missing entirely.
    if expose_subs == 0:
        text = text.rstrip() + f"\nEXPOSE {FORGE_APP_PORT}\n"

    # Rewrite any `--port N` (in any quoting / spacing form) to the canonical port.
    text = _PORT_FLAG_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{FORGE_APP_PORT}",
        text,
    )

    if text != original:
        dockerfile.write_text(text)
        return True
    return False


def normalise_dockerfile_base(dockerfile: Path) -> bool:
    """Rewrite any `FROM python:*` line to `FROM <FORGE_PYTHON_BASE>`.

    Returns True if the file was modified. The LLM that generates the
    Dockerfile picks an arbitrary python tag (e.g. 3.11-slim, 3.12-bookworm),
    each requiring its own Hub pull. Normalising to a single canonical
    base means we only need to keep ONE image warm locally.
    """
    original = dockerfile.read_text()
    new_lines: list[str] = []
    changed = False
    for line in original.splitlines(keepends=True):
        m = _PYTHON_FROM_RE.match(line.rstrip("\r\n"))
        if m:
            stage_suffix = m.group(2) or ""
            ending = "\n" if line.endswith("\n") else ""
            replacement = f"{m.group(1)}{FORGE_PYTHON_BASE}{stage_suffix}{ending}"
            if replacement != line:
                changed = True
                new_lines.append(replacement)
                continue
        new_lines.append(line)
    if changed:
        dockerfile.write_text("".join(new_lines))
    return changed


def parse_from_image(dockerfile: Path) -> str | None:
    """Return the base image name from the first FROM line in a Dockerfile."""
    for line in dockerfile.read_text().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            parts = stripped.split()
            if len(parts) >= 2:
                image = parts[1]
                # Strip build-stage alias (e.g. "python:3.12-slim AS builder")
                return image if image.lower() != "scratch" else None
    return None
