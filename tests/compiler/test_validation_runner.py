import tempfile
import sys
from pathlib import Path
from forge.compiler.package_builder import PackageBuilder
from forge.compiler.validation_runner import ValidationRunner, ValidationResult
from forge.extraction.schemas import (
    CompilerInput, EntityDef, FieldDef, ActionDef, ActionParam,
    TaskTemplate, SuccessCondition,
)


def _counter_input() -> CompilerInput:
    return CompilerInput(
        project_name="counter_env",
        domain="counter",
        entities=[
            EntityDef(name="counter", fields=[
                FieldDef(name="id", type="string"),
                FieldDef(name="value", type="integer", default=0),
            ])
        ],
        actions=[
            ActionDef(
                name="increment",
                params=[ActionParam(name="counter_id", type="string")],
                mutates=["counter.value"],
            )
        ],
        tasks=[
            TaskTemplate(
                name="reach_target",
                description="Reach target value",
                success_conditions=[
                    SuccessCondition(type="state_check", expression="counter.value >= target")
                ],
            )
        ],
    )


def test_validation_runner_returns_validation_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        ci = _counter_input()
        pkg_dir = PackageBuilder(output_dir).build(ci)
        runner = ValidationRunner(generated_envs_root=output_dir)
        result = runner.run(pkg_dir)
        assert isinstance(result, ValidationResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.output, str)


def test_validation_result_has_test_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        ci = _counter_input()
        pkg_dir = PackageBuilder(output_dir).build(ci)
        runner = ValidationRunner(generated_envs_root=output_dir)
        result = runner.run(pkg_dir)
        assert result.total_tests >= 0
