import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from forge.benchmark.data_collector import DataCollector, CollectionConfig, CollectionCheckpoint


def test_checkpoint_saves_and_loads(tmp_path):
    ckpt = CollectionCheckpoint(output_dir=tmp_path)
    ckpt.mark_done(domain="email", task_name="email_read_star", seed=0)
    ckpt.mark_done(domain="email", task_name="email_read_star", seed=1)
    ckpt2 = CollectionCheckpoint(output_dir=tmp_path)
    assert ckpt2.is_done(domain="email", task_name="email_read_star", seed=0)
    assert ckpt2.is_done(domain="email", task_name="email_read_star", seed=1)
    assert not ckpt2.is_done(domain="email", task_name="email_read_star", seed=2)


def test_checkpoint_skips_completed(tmp_path):
    ckpt = CollectionCheckpoint(output_dir=tmp_path)
    ckpt.mark_done("email", "email_read_star", 0)
    assert ckpt.is_done("email", "email_read_star", 0)
    assert not ckpt.is_done("email", "email_reply_label", 0)


def test_collection_config_defaults():
    cfg = CollectionConfig(domains=["email"], depth=3, seeds=2, output_dir=Path("/tmp/bench"))
    assert cfg.domains == ["email"]
    assert cfg.depth == 3
    assert cfg.seeds == 2


def test_collector_skips_done_tasks(tmp_path):
    cfg = CollectionConfig(domains=["email"], depth=1, seeds=2, output_dir=tmp_path)
    ckpt = CollectionCheckpoint(output_dir=tmp_path)
    ckpt.mark_done("email", "email_read_star", 0)
    ckpt.mark_done("email", "email_read_star", 1)

    collector = DataCollector(cfg)
    runs = list(collector._pending_runs(ckpt))
    # All seeds for email_read_star are done — nothing to run
    assert not any(
        r["domain"] == "email" and r["task_name"] == "email_read_star"
        for r in runs
    )
