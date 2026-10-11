"""`forge train` CLI command redirects to Autotune."""
from pathlib import Path
from typer.testing import CliRunner
from forge.cli.main import app

runner = CliRunner()


def test_train_redirects_to_autotune(tmp_path: Path):
    result = runner.invoke(app, ["train", "--data", str(tmp_path)])
    assert result.exit_code == 0
    assert "autotune" in result.output.lower()
    assert "https://github.com/mostofashakib/Autotune" in result.output


def test_train_missing_data_flag_rejected():
    """Negative case: missing required --data option is rejected."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
    assert "error" in result.output.lower() or "missing" in result.output.lower()


def test_train_rejects_extra_unexpected_argument(tmp_path: Path):
    """False-positive case: look-valid invocation with extra unexpected argument is rejected."""
    result = runner.invoke(app, ["train", "--data", str(tmp_path), "unexpected_positional_arg"])
    assert result.exit_code != 0
