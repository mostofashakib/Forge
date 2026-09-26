"""The Python dependency set every environment container installs."""
from __future__ import annotations

import re
from pathlib import Path

# Fully pinned, with hashes, so pip resolves nothing at build time.
# Regenerate it from runtime_requirements.in (the command is in that file).
RUNTIME_LOCK = Path(__file__).with_name("runtime_requirements.lock")

# Projects whose import name differs from their name on PyPI.
_IMPORT_NAMES: dict[str, tuple[str, ...]] = {
    "python-multipart": ("multipart", "python_multipart"),
    "python-dotenv": ("dotenv",),
    "pyyaml": ("yaml",),
}
_PINNED_PROJECT_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==", re.MULTILINE)


def locked_import_names() -> frozenset[str]:
    """Top-level module names that code inside a container can import."""
    names: set[str] = set()
    for project in _PINNED_PROJECT_RE.findall(RUNTIME_LOCK.read_text()):
        key = project.lower()
        names.update(_IMPORT_NAMES.get(key, (key.replace("-", "_"),)))
    return frozenset(names)
