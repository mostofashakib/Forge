import tempfile
from pathlib import Path
from forge.compiler.override_validator import OverrideValidator
from forge.customization.hooks import clear_registry
from forge.extraction.schemas import (
    CompilerInput, EntityDef, FieldDef, ActionDef, ActionParam,
    TaskTemplate, SuccessCondition,
)


def _counter_input() -> CompilerInput:
    return CompilerInput(
        project_name="counter_env",
        domain="counter",
        entities=[EntityDef(name="counter", fields=[FieldDef(name="id", type="string")])],
        actions=[ActionDef(name="increment", params=[ActionParam(name="counter_id", type="string")])],
        tasks=[TaskTemplate(
            name="reach_target",
            description="Reach target",
            success_conditions=[SuccessCondition(type="state_check", expression="done")],
        )],
    )


def test_validator_passes_when_no_custom_dir(tmp_path):
    clear_registry()
    result = OverrideValidator().validate(tmp_path, _counter_input())
    assert result.valid
    assert result.errors == []


def test_validator_passes_when_overrides_match(tmp_path):
    clear_registry()
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "transitions.py").write_text(
        "from forge.customization.hooks import override_transition\n"
        "@override_transition('increment')\n"
        "def my_fn(s, a, c): pass\n"
    )
    result = OverrideValidator().validate(tmp_path, _counter_input())
    assert result.valid
    assert result.errors == []


def test_validator_fails_when_action_deleted(tmp_path):
    clear_registry()
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "transitions.py").write_text(
        "from forge.customization.hooks import override_transition\n"
        "@override_transition('old_action_deleted')\n"
        "def my_fn(s, a, c): pass\n"
    )
    result = OverrideValidator().validate(tmp_path, _counter_input())
    assert not result.valid
    assert any("old_action_deleted" in e for e in result.errors)


def test_validator_fails_when_task_deleted(tmp_path):
    clear_registry()
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "verifiers.py").write_text(
        "from forge.customization.hooks import verifier\n"
        "@verifier('old_task_gone')\n"
        "def my_fn(s, t, task): pass\n"
    )
    result = OverrideValidator().validate(tmp_path, _counter_input())
    assert not result.valid
    assert any("old_task_gone" in e for e in result.errors)


def test_validator_fails_for_stale_reward_override(tmp_path):
    clear_registry()
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "rewards.py").write_text(
        "from forge.customization.hooks import reward\n"
        "@reward('nonexistent_task')\n"
        "def my_fn(s, t, vr, task=None): pass\n"
    )
    result = OverrideValidator().validate(tmp_path, _counter_input())
    assert not result.valid
    assert any("nonexistent_task" in e for e in result.errors)
