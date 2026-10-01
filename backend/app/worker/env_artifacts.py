"""Read the files a generated environment keeps beside its app.

Both loaders degrade to None with a logged warning, so one malformed file
turns a run into a plainer run instead of failing every episode.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_manifest(env_dir: Path):
    """The env's state schema manifest, or None when it has none or it is unreadable."""
    from forge.schema.state_schema import StateSchemaManifest

    manifest_path = env_dir / "state_schema.json"
    if not manifest_path.exists():
        return None
    try:
        return StateSchemaManifest.model_validate_json(manifest_path.read_text())
    except Exception as exc:
        logger.warning("[manifest] could not load %s: %s", manifest_path, exc)
        return None


def load_personas(env_dir: Path):
    """The cast configured for this environment, or None if it has none.

    Read here rather than inside the runner so a malformed `personas:` block
    degrades an agent run to an empty environment with a logged warning,
    instead of failing every episode of the run.
    """
    config_path = env_dir / "custom" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml

        from forge.personas.config import load_population

        raw = yaml.safe_load(config_path.read_text()) or {}
        population = load_population(raw.get("personas"))
    except Exception as exc:
        logger.warning("[personas] could not load cast for %s: %s", env_dir.name, exc)
        return None
    return population if population.enabled else None
