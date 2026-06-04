from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from forge.benchmark.task_suite import TaskSuite

logger = logging.getLogger(__name__)

_CHECKPOINT_FILE = "checkpoint.json"


@dataclass
class CollectionConfig:
    domains: list[str]
    depth: int
    seeds: int
    output_dir: Path


class CollectionCheckpoint:
    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / _CHECKPOINT_FILE
        self._done: set[str] = set()
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._done = set(data.get("done", []))
            except Exception:
                pass

    def _key(self, domain: str, task_name: str, seed: int) -> str:
        return f"{domain}::{task_name}::{seed}"

    def is_done(self, domain: str, task_name: str, seed: int) -> bool:
        return self._key(domain, task_name, seed) in self._done

    def mark_done(self, domain: str, task_name: str, seed: int) -> None:
        self._done.add(self._key(domain, task_name, seed))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"done": sorted(self._done)}))


class DataCollector:
    def __init__(self, config: CollectionConfig) -> None:
        self._cfg = config
        self._suite = TaskSuite()

    def _pending_runs(self, checkpoint: CollectionCheckpoint) -> list[dict]:
        runs = []
        for domain in self._cfg.domains:
            tasks = self._suite.tasks_for(domain=domain, depth=self._cfg.depth)
            for task in tasks:
                for seed in range(self._cfg.seeds):
                    if not checkpoint.is_done(domain, task.name, seed):
                        runs.append({
                            "domain": domain,
                            "task_name": task.name,
                            "task": task,
                            "seed": seed,
                        })
        return runs

    def collect(self, episode_runner_fn) -> None:
        """Run pending episodes using episode_runner_fn(task, seed, output_path).

        episode_runner_fn signature:
            (task: Task, seed: int, jsonl_path: Path) -> None

        Checkpoints after each completed seed. Re-running collect() skips done episodes.
        """
        output_dir = self._cfg.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = CollectionCheckpoint(output_dir=output_dir)
        pending = self._pending_runs(checkpoint)
        logger.info("[collector] %d episodes pending", len(pending))

        for run in pending:
            task = run["task"]
            seed = run["seed"]
            domain = run["domain"]
            ep_dir = output_dir / domain / task.name / f"seed_{seed}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = ep_dir / "episode.jsonl"

            try:
                episode_runner_fn(task, seed, jsonl_path)
                checkpoint.mark_done(domain, task.name, seed)
                logger.info("[collector] done: %s seed=%d", task.name, seed)
            except Exception as exc:
                logger.error("[collector] failed: %s seed=%d: %s", task.name, seed, exc)
