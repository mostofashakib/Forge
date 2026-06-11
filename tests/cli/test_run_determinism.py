# tests/cli/test_run_determinism.py
import sys
import textwrap
import pytest
from typer.testing import CliRunner
from forge.cli.main import app

runner = CliRunner()

_WRAPPER = textwrap.dedent("""
    import copy
    from forge.runtime.env import ForgeEnv
    from forge.runtime.reward import RewardEngine
    from forge.runtime.snapshot import EnvironmentSpec
    from forge.runtime.transition import TransitionEngine, TransitionResult
    from forge.runtime.verifier import VerifierEngine


    class Factory:
        def create(self, ctx, options):
            return {"counter": {"c_0": {"id": "c_0", "value": INITIAL_VALUE}}}


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


def _write_env(root, name: str, initial_value: str) -> None:
    pkg = root / "generated_envs" / name
    pkg.mkdir(parents=True)
    (root / "generated_envs" / "__init__.py").touch()
    (pkg / "__init__.py").touch()
    source = _WRAPPER.replace("ENVNAME", name).replace("INITIAL_VALUE", initial_value)
    (pkg / "gym_wrapper.py").write_text(source)


@pytest.fixture
def cli_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FORGE_SKIP_DETERMINISM_CHECK", raising=False)
    for mod in [m for m in sys.modules if m.startswith("generated_envs")]:
        del sys.modules[mod]
    sys.path.insert(0, str(tmp_path))
    yield tmp_path
    sys.path.remove(str(tmp_path))


def test_run_reports_determinism_check_passed(cli_root):
    _write_env(cli_root, "cli_det_ok", "ctx.rng.randint(0, 1000)")
    result = runner.invoke(app, ["run", "--env", "cli_det_ok", "--steps", "3"])
    assert result.exit_code == 0
    assert "determinism check passed" in result.output.lower()


def test_run_fails_on_nondeterministic_env(cli_root):
    _write_env(cli_root, "cli_det_bad", "__import__('time').time_ns()")
    result = runner.invoke(app, ["run", "--env", "cli_det_bad", "--steps", "3"])
    assert result.exit_code != 0
    assert "not deterministic" in result.output.lower()


def test_run_skips_check_when_env_var_set(cli_root, monkeypatch):
    _write_env(cli_root, "cli_det_skip", "__import__('time').time_ns()")
    monkeypatch.setenv("FORGE_SKIP_DETERMINISM_CHECK", "1")
    result = runner.invoke(app, ["run", "--env", "cli_det_skip", "--steps", "3"])
    assert result.exit_code == 0


def test_export_fails_on_nondeterministic_env(cli_root):
    _write_env(cli_root, "cli_det_bad_exp", "__import__('time').time_ns()")
    result = runner.invoke(
        app, ["export", "--env", "cli_det_bad_exp", "--steps", "3", "--out", str(cli_root / "exports")]
    )
    assert result.exit_code != 0
    assert "not deterministic" in result.output.lower()
