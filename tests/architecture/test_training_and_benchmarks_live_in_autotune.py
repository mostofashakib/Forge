"""Training and model benchmarking live in Autotune. Forge exposes neither."""
from typer.testing import CliRunner

from backend.app.main import app as api_app
from forge.cli.main import app as cli_app


def _api_paths() -> set[str]:
    return set(api_app.openapi()["paths"])


def test_the_api_has_no_training_routes():
    assert not [path for path in _api_paths() if path.startswith("/api/training")]


def test_the_api_has_no_benchmark_routes():
    assert not [path for path in _api_paths() if path.startswith("/api/benchmark")]


def test_the_api_has_no_model_serving_routes():
    serving = {"/api/distributed/hardware", "/api/distributed/inference", "/api/distributed/inference/batch"}
    assert not serving & _api_paths()


def test_the_api_has_no_distributed_queue_or_scheduler_routes():
    # Nothing consumed the in-memory queue, and /schedule reported builds it never ran.
    assert not [path for path in _api_paths() if path.startswith("/api/distributed")]


def test_the_cli_has_no_train_command():
    result = CliRunner().invoke(cli_app, ["train", "--help"])
    assert result.exit_code != 0
    assert "No such command" in result.output


def test_the_cli_has_no_benchmark_command():
    result = CliRunner().invoke(cli_app, ["benchmark", "--help"])
    assert result.exit_code != 0
    assert "No such command" in result.output


def test_forge_declares_no_model_inference_contracts():
    import importlib.util

    import forge.contracts

    assert importlib.util.find_spec("forge.contracts.inference") is None
    assert not [name for name in forge.contracts.__all__ if name.startswith("Inference")]
