import tempfile
import pytest
from pathlib import Path
from forge.compiler.package_builder import PackageBuilder
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


def test_package_builder_creates_expected_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        builder = PackageBuilder(output_root=output_dir)
        pkg_dir = builder.build(_counter_input())

        assert (pkg_dir / "__init__.py").exists()
        assert (pkg_dir / "state_models.py").exists()
        assert (pkg_dir / "action_models.py").exists()
        assert (pkg_dir / "initial_state.py").exists()
        assert (pkg_dir / "gym_wrapper.py").exists()
        assert (pkg_dir / "transitions" / "__init__.py").exists()
        assert (pkg_dir / "transitions" / "increment.py").exists()
        assert (pkg_dir / "verifiers" / "__init__.py").exists()
        assert (pkg_dir / "verifiers" / "reach_target.py").exists()
        assert (pkg_dir / "rewards" / "__init__.py").exists()
        assert (pkg_dir / "rewards" / "reach_target.py").exists()
        assert (pkg_dir / "tests" / "__init__.py").exists()
        assert (pkg_dir / "tests" / "test_determinism.py").exists()


def test_package_builder_custom_dir_not_overwritten_on_rebuild():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        builder = PackageBuilder(output_root=output_dir)
        ci = _counter_input()
        pkg_dir = builder.build(ci)

        custom_file = pkg_dir / "custom" / "transitions.py"
        custom_file.write_text("# my custom code\n")

        builder.build(ci)
        assert custom_file.read_text() == "# my custom code\n"


def test_package_builder_returns_correct_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        builder = PackageBuilder(output_root=output_dir)
        pkg_dir = builder.build(_counter_input())
        assert pkg_dir.name == "counter_env"


def test_compiler_input_rejects_path_traversal_project_name():
    data = _counter_input().model_dump()
    data["project_name"] = "../../outside"
    with pytest.raises(ValueError):
        CompilerInput.model_validate(data)


def test_compiler_input_rejects_path_traversal_action_name():
    data = _counter_input().model_dump()
    data["actions"][0]["name"] = "../escape"
    with pytest.raises(ValueError):
        CompilerInput.model_validate(data)
