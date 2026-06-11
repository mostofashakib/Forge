# tests/backend/test_env_loader_determinism.py
import sys
import textwrap
import pytest
from backend.app.utils.env_loader import load_forge_env
from forge.runtime.determinism import DeterminismError

_DETERMINISTIC_WRAPPER = textwrap.dedent("""
    import copy
    from forge.runtime.env import ForgeEnv
    from forge.runtime.reward import RewardEngine
    from forge.runtime.snapshot import EnvironmentSpec
    from forge.runtime.transition import TransitionEngine, TransitionResult
    from forge.runtime.verifier import VerifierEngine


    class Factory:
        def create(self, ctx, options):
            return {"counter": {"c_0": {"id": "c_0", "value": ctx.rng.randint(0, 1000)}}}


    def increment(state, action, ctx):
        new_state = copy.deepcopy(state)
        new_state["counter"]["c_0"]["value"] += 1
        return TransitionResult(state=new_state, events=[])


    def build_ENVNAME_env(max_steps: int = 10) -> ForgeEnv:
        te = TransitionEngine()
        te.register("increment", increment)
        return ForgeEnv(
            env_spec=EnvironmentSpec(name="ENVNAME", domain="test", max_steps=max_steps),
            initial_state_factory=Factory(),
            transition_engine=te,
            verifier_engine=VerifierEngine(),
            reward_engine=RewardEngine(),
        )
""")

_NONDETERMINISTIC_WRAPPER = _DETERMINISTIC_WRAPPER.replace(
    'return {"counter": {"c_0": {"id": "c_0", "value": ctx.rng.randint(0, 1000)}}}',
    'import time; return {"counter": {"c_0": {"id": "c_0", "value": time.time_ns()}}}',
)


def _write_env(tmp_path, name: str, source: str) -> None:
    pkg = tmp_path / "generated_envs" / name
    pkg.mkdir(parents=True)
    (tmp_path / "generated_envs" / "__init__.py").touch()
    (pkg / "__init__.py").touch()
    (pkg / "gym_wrapper.py").write_text(source.replace("ENVNAME", name))


@pytest.fixture
def envs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_GENERATED_ENVS_DIR", str(tmp_path / "generated_envs"))
    monkeypatch.delenv("FORGE_SKIP_DETERMINISM_CHECK", raising=False)
    for mod in [m for m in sys.modules if m.startswith("generated_envs")]:
        del sys.modules[mod]
    return tmp_path


def test_load_forge_env_runs_determinism_check_and_passes(envs_dir):
    _write_env(envs_dir, "det_ok_env", _DETERMINISTIC_WRAPPER)
    env = load_forge_env("det_ok_env", telemetry=None)
    assert env is not None


def test_load_forge_env_raises_on_nondeterministic_env(envs_dir):
    _write_env(envs_dir, "det_bad_env", _NONDETERMINISTIC_WRAPPER)
    with pytest.raises(DeterminismError):
        load_forge_env("det_bad_env", telemetry=None)


def test_load_forge_env_skips_check_when_env_var_set(envs_dir, monkeypatch):
    _write_env(envs_dir, "det_skip_env", _NONDETERMINISTIC_WRAPPER)
    monkeypatch.setenv("FORGE_SKIP_DETERMINISM_CHECK", "1")
    env = load_forge_env("det_skip_env", telemetry=None)
    assert env is not None
