import tempfile
from pathlib import Path
from forge.customization.hooks import clear_registry
from forge.customization.loader import CustomizationLoader
from forge.runtime.reward import RewardEngine
from forge.runtime.transition import TransitionEngine
from forge.runtime.verifier import VerifierEngine


def _make_engines():
    te = TransitionEngine()
    ve = VerifierEngine()
    re = RewardEngine()
    return te, ve, re


def test_loader_applies_transition_override(tmp_path):
    clear_registry()
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "transitions.py").write_text(
        "from forge.customization.hooks import override_transition\n"
        "@override_transition('my_action')\n"
        "def custom_fn(state, action, ctx):\n"
        "    return None\n"
    )
    te, ve, re = _make_engines()
    te.register("my_action", lambda s, a, c: None)  # original handler

    CustomizationLoader(tmp_path).apply(te, ve, re)

    handler = te._handlers["my_action"]
    assert handler.__name__ == "custom_fn"


def test_loader_applies_verifier_override(tmp_path):
    clear_registry()
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "verifiers.py").write_text(
        "from forge.customization.hooks import verifier\n"
        "@verifier('my_task')\n"
        "def custom_verifier(state, traj, task):\n"
        "    return None\n"
    )
    te, ve, re = _make_engines()
    CustomizationLoader(tmp_path).apply(te, ve, re)
    assert "my_task" in ve._verifiers


def test_loader_applies_reward_override(tmp_path):
    clear_registry()
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "rewards.py").write_text(
        "from forge.customization.hooks import reward\n"
        "@reward('my_task')\n"
        "def custom_reward(state, traj, vr, task=None):\n"
        "    return None\n"
    )
    te, ve, re = _make_engines()
    CustomizationLoader(tmp_path).apply(te, ve, re)
    assert "my_task" in re._task_fns


def test_loader_skips_missing_custom_dir(tmp_path):
    te, ve, re = _make_engines()
    # Should not raise even if custom/ does not exist
    CustomizationLoader(tmp_path).apply(te, ve, re)


def test_loader_skips_dunder_files(tmp_path):
    clear_registry()
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "__init__.py").write_text(
        "raise RuntimeError('should not be imported')\n"
    )
    te, ve, re = _make_engines()
    CustomizationLoader(tmp_path).apply(te, ve, re)  # must not raise


def test_loader_clears_registry_before_loading(tmp_path):
    from forge.customization.hooks import _registry
    _registry["transitions"]["stale_action"] = lambda: None

    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "transitions.py").write_text(
        "from forge.customization.hooks import override_transition\n"
        "@override_transition('fresh_action')\n"
        "def fn(s, a, c): pass\n"
    )
    te, ve, re = _make_engines()
    CustomizationLoader(tmp_path).apply(te, ve, re)

    # stale_action was in registry before load — after clear it should NOT be applied
    assert "stale_action" not in te._handlers
    assert "fresh_action" in te._handlers
