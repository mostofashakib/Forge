"""Policy training lives in Autotune. Forge exposes no training surface."""
from typer.testing import CliRunner

from backend.app.main import app as api_app
from forge.cli.main import app as cli_app


def _api_paths() -> set[str]:
    return set(api_app.openapi()["paths"])


def test_the_api_has_no_training_routes():
    assert not [path for path in _api_paths() if path.startswith("/api/training")]


def test_the_api_has_no_transfer_benchmark_routes():
    assert not [path for path in _api_paths() if path.startswith("/api/benchmark/transfer")]


def test_the_api_keeps_held_out_evaluation():
    assert "/api/benchmark/evals" in _api_paths()


def test_the_cli_has_no_train_command():
    result = CliRunner().invoke(cli_app, ["train", "--help"])
    assert result.exit_code != 0
    assert "No such command" in result.output


def test_the_cli_has_no_transfer_benchmark_command():
    result = CliRunner().invoke(cli_app, ["benchmark", "transfer", "--help"])
    assert result.exit_code != 0
    assert "No such command" in result.output
