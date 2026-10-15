from __future__ import annotations

import keyword
import re
from pathlib import Path


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_identifier(value: str, *, label: str = "identifier") -> str:
    """Validate names that become Python symbols, module names, or filenames."""
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{label} must be a valid Python identifier containing only letters, "
            "digits, and underscores"
        )
    return value


def module_name_for(env_name: str) -> str:
    """The Python package name for an environment's generated code.

    Environment names may hold hyphens and start with a digit, and generated
    code imports from this package, so hyphens become underscores, a leading
    digit gets an env_ prefix, and a Python keyword gets an _env suffix.
    """
    module = validate_path_segment(env_name, label="env_name").replace("-", "_")
    if module[0].isdigit():
        module = f"env_{module}"
    if keyword.iskeyword(module):
        module = f"{module}_env"
    return module


def validate_path_segment(value: str, *, label: str = "name") -> str:
    """Validate a user-facing name that becomes exactly one directory segment."""
    if not _PATH_SEGMENT_RE.fullmatch(value):
        raise ValueError(
            f"{label} must contain only letters, digits, underscores, and hyphens"
        )
    return value


def confined_path(root: Path, *parts: str | Path) -> Path:
    """Return a path below ``root`` or reject absolute/path-traversal input."""
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"Path escapes configured root: {candidate}")
    return candidate


def confined_relative_path(root: Path, relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("Absolute paths are not allowed")
    return confined_path(root, relative)
