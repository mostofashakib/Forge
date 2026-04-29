# forge/runtime/clustering.py
from __future__ import annotations
import json
from dataclasses import dataclass, field
from forge.runtime.replay import EpisodeRecord


@dataclass
class FailureCluster:
    check_name: str
    count: int
    episode_ids: list[str] = field(default_factory=list)


class FailureClusterer:
    def cluster(self, episodes: list[EpisodeRecord]) -> list[FailureCluster]:
        buckets: dict[str, list[str]] = {}
        for record in episodes:
            if record.episode.passed:
                continue
            check_name = self._first_failed_check(record)
            if check_name is None:
                continue
            buckets.setdefault(check_name, []).append(record.episode.id)

        clusters = [
            FailureCluster(
                check_name=name,
                count=len(ids),
                episode_ids=ids[:5],
            )
            for name, ids in buckets.items()
        ]
        clusters.sort(key=lambda c: c.count, reverse=True)
        return clusters[:5]

    def _first_failed_check(self, record: EpisodeRecord) -> str | None:
        for step in record.steps:
            try:
                results = json.loads(step.verifier_results)
            except (json.JSONDecodeError, TypeError):
                continue
            for vr in results:
                for check in vr.get("checks", []):
                    if not check.get("passed", True):
                        return check.get("name")
        return None
