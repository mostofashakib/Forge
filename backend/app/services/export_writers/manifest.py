"""The manifest every dataset export writes beside its files."""
from __future__ import annotations

from datetime import datetime, timezone

MANIFEST_FILE = "dataset_manifest.json"
MANIFEST_VERSION = "1.0.0"


def dataset_manifest(*, files: list[str], safety: dict, **fields) -> dict:
    """The shared manifest fields, plus what this kind of export records about itself."""
    return {
        "manifest_version": MANIFEST_VERSION,
        **fields,
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "safety": safety,
    }
