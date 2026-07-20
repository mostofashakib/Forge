from pathlib import Path
from forge.benchmark.compiled_tasks import CompiledTaskProvider
from forge.benchmark.data_collector import DataCollector, CollectionConfig, CollectionCheckpoint
from forge.extraction.schemas import CompilerInput, SuccessCondition, TaskTemplate


def _provider_with(env_name: str, task_names: list[str]) -> CompiledTaskProvider:
    ci = CompilerInput(
        project_name=env_name,
        domain="generated",
        entities=[],
        actions=[],
        tasks=[
            TaskTemplate(
                name=name,
                description=f"Objective for {name}",
                success_conditions=[SuccessCondition(type="state_check", expression="x == 1")],
            )
            for name in task_names
        ],
    )
    return CompiledTaskProvider(loader=lambda name: ci if name == env_name else None)


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
    cfg = CollectionConfig(domains=["crm_env"], depth=1, seeds=2, output_dir=tmp_path)
    ckpt = CollectionCheckpoint(output_dir=tmp_path)
    ckpt.mark_done("crm_env", "close_ticket", 0)
    ckpt.mark_done("crm_env", "close_ticket", 1)

    collector = DataCollector(cfg, task_provider=_provider_with("crm_env", ["close_ticket"]))
    runs = list(collector._pending_runs(ckpt))
    # All seeds for close_ticket are done — nothing to run
    assert not any(
        r["domain"] == "crm_env" and r["task_name"] == "close_ticket"
        for r in runs
    )
