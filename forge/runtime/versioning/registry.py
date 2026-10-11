"""Artifact Version Registry.

Computes cryptographic SHA-256 hashes and tracks version provenance for
prompts, verifiers, datasets, and initial test suites to allow exact
reproducibility and deep debugging across agent rollouts.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from forge.config import get_config
from forge.contracts.versioning import (
    ArtifactType,
    InitialTestSuite,
    VersionedArtifact,
    VersionProvenance,
)


class ArtifactVersionRegistry:
    """Registry maintaining cryptographically hashed versions of evaluation artifacts."""

    def __init__(self) -> None:
        self._artifacts: dict[tuple[ArtifactType, str], VersionedArtifact] = {}

    @staticmethod
    def compute_hash(content: Any) -> str:
        """Compute a deterministic, canonical SHA-256 hash for any artifact payload."""
        if isinstance(content, str):
            canonical = content.strip().encode("utf-8")
        elif isinstance(content, (dict, list)):
            canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
        elif hasattr(content, "model_dump"):
            canonical = json.dumps(content.model_dump(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        elif hasattr(content, "__dict__"):
            # Extract plain public dictionary
            public_dict = {k: v for k, v in content.__dict__.items() if not k.startswith("_")}
            canonical = json.dumps(public_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        else:
            canonical = str(content).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def register(
        self,
        artifact_type: ArtifactType,
        name: str,
        content: Any,
        version_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VersionedArtifact:
        """Register a versioned artifact and index by (type, name)."""
        vhash = self.compute_hash(content)
        tag = version_tag or f"v{vhash[:8]}"
        artifact = VersionedArtifact(
            artifact_type=artifact_type,
            name=name,
            version_hash=vhash,
            version_tag=tag,
            metadata=metadata or {},
        )
        self._artifacts[(artifact_type, name)] = artifact
        return artifact

    def register_prompt(
        self,
        name: str,
        template: Any,
        version_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VersionedArtifact:
        """Register a versioned prompt template."""
        tag = version_tag or get_config().versions.default_prompt_version
        return self.register(ArtifactType.PROMPT, name, template, tag, metadata)

    def register_verifier(
        self,
        name: str,
        verifier_or_spec: Any,
        version_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VersionedArtifact:
        """Register a versioned verifier or verification specification."""
        tag = version_tag or get_config().versions.default_verifier_version
        return self.register(ArtifactType.VERIFIER, name, verifier_or_spec, tag, metadata)

    def register_data(
        self,
        name: str,
        dataset_or_task: Any,
        version_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VersionedArtifact:
        """Register a versioned dataset, task, or environment initial universe."""
        tag = version_tag or get_config().versions.default_data_version
        return self.register(ArtifactType.DATA, name, dataset_or_task, tag, metadata)

    def register_test_suite(
        self,
        name: str,
        test_suite: InitialTestSuite | Any,
        version_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VersionedArtifact:
        """Register a versioned authoritative initial test suite."""
        tag = version_tag or get_config().versions.default_test_suite_version
        return self.register(ArtifactType.TEST_SUITE, name, test_suite, tag, metadata)

    def get_artifact(self, artifact_type: ArtifactType, name: str) -> VersionedArtifact | None:
        """Retrieve registered artifact by type and name."""
        return self._artifacts.get((artifact_type, name))

    def create_provenance(
        self,
        prompt: Any,
        verifier: Any,
        data: Any,
        test_suite: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VersionProvenance:
        """Construct an immutable provenance record from artifact names or direct payloads."""
        p_hash = self.compute_hash(prompt)
        v_hash = self.compute_hash(verifier)
        d_hash = self.compute_hash(data)
        t_hash = self.compute_hash(test_suite) if test_suite is not None else None

        composite_source = f"{p_hash}:{v_hash}:{d_hash}:{t_hash or ''}".encode("utf-8")
        composite_digest = hashlib.sha256(composite_source).hexdigest()

        now_iso = datetime.now(timezone.utc).isoformat()
        return VersionProvenance(
            prompt_version=p_hash[:16],
            verifier_version=v_hash[:16],
            data_version=d_hash[:16],
            test_suite_version=t_hash[:16] if t_hash else None,
            composite_digest=composite_digest[:16],
            created_at=now_iso,
            metadata=metadata or {},
        )

    def diff_versions(self, a: VersionedArtifact, b: VersionedArtifact) -> dict[str, Any]:
        """Produce a diagnostic diff between two artifact versions."""
        return {
            "name": a.name,
            "type": a.artifact_type.value,
            "version_a": {"tag": a.version_tag, "hash": a.version_hash},
            "version_b": {"tag": b.version_tag, "hash": b.version_hash},
            "identical": a.version_hash == b.version_hash,
            "metadata_diff": {
                "added_in_b": {k: b.metadata[k] for k in b.metadata if k not in a.metadata},
                "removed_in_b": {k: a.metadata[k] for k in a.metadata if k not in b.metadata},
                "changed": {
                    k: {"from": a.metadata[k], "to": b.metadata[k]}
                    for k in a.metadata
                    if k in b.metadata and a.metadata[k] != b.metadata[k]
                },
            },
        }
