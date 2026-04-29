import json
import tempfile
from pathlib import Path
import pytest
from typer.testing import CliRunner
from forge.cli.main import app
from forge.extraction.schemas import (
    CompilerInput, EntityDef, FieldDef, ActionDef, ActionParam,
    TaskTemplate, SuccessCondition,
)

runner = CliRunner()


def _counter_input() -> CompilerInput:
    return CompilerInput(
        project_name="counter_env",
        domain="counter",
        entities=[EntityDef(name="counter", fields=[
            FieldDef(name="id", type="string"),
            FieldDef(name="value", type="integer", default=0),
        ])],
        actions=[ActionDef(
            name="increment",
            params=[ActionParam(name="counter_id", type="string")],
            mutates=["counter.value"],
        )],
        tasks=[TaskTemplate(
            name="reach_target",
            description="Reach target",
            success_conditions=[SuccessCondition(type="state_check", expression="done")],
        )],
    )


def test_cli_has_compile_command():
    result = runner.invoke(app, ["compile", "--help"])
    assert result.exit_code == 0
    assert "compile" in result.output.lower() or "input" in result.output.lower()


def test_cli_has_validate_command():
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0


def test_cli_has_run_command():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_cli_has_export_command():
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0


def test_compile_command_builds_package(tmp_path):
    ci = _counter_input()
    input_file = tmp_path / "compiler_input.json"
    input_file.write_text(ci.model_dump_json())
    output_dir = tmp_path / "out"

    result = runner.invoke(app, [
        "compile",
        "--input", str(input_file),
        "--output", str(output_dir),
    ])
    assert result.exit_code == 0, result.output
    assert (output_dir / "counter_env" / "gym_wrapper.py").exists()


def test_compile_command_fails_on_invalid_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json {{{")
    result = runner.invoke(app, ["compile", "--input", str(bad_file), "--output", str(tmp_path)])
    assert result.exit_code != 0


def test_validate_command_on_built_package(tmp_path):
    from forge.compiler.package_builder import PackageBuilder
    ci = _counter_input()
    pkg_dir = PackageBuilder(tmp_path).build(ci)

    result = runner.invoke(app, ["validate", str(pkg_dir)])
    assert result.exit_code == 0, result.output
    assert "passed" in result.output.lower() or "valid" in result.output.lower()


def test_validate_command_fails_on_nonexistent_dir(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "does_not_exist")])
    assert result.exit_code != 0
