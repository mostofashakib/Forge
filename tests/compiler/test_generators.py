import ast
import pytest
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


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def test_state_model_generator_produces_valid_python():
    from forge.compiler.generators.state_model import StateModelGenerator
    code = StateModelGenerator().generate(_counter_input())
    assert _is_valid_python(code), f"Invalid Python:\n{code}"
    assert "class Counter(BaseModel)" in code
    assert "value: int = 0" in code


def test_action_schema_generator_produces_valid_python():
    from forge.compiler.generators.action_schema import ActionSchemaGenerator
    code = ActionSchemaGenerator().generate(_counter_input())
    assert _is_valid_python(code), f"Invalid Python:\n{code}"
    assert "class IncrementAction(TypedDict)" in code
    assert "counter_id: str" in code


def test_transition_generator_produces_one_file_per_action():
    from forge.compiler.generators.transition import TransitionGenerator
    files = TransitionGenerator().generate(_counter_input())
    assert "increment" in files
    code = files["increment"]
    assert _is_valid_python(code), f"Invalid Python:\n{code}"
    assert "def apply_increment(" in code
    assert "copy.deepcopy(state)" in code


def test_verifier_generator_produces_one_file_per_task():
    from forge.compiler.generators.verifier import VerifierGenerator
    files = VerifierGenerator().generate(_counter_input())
    assert "reach_target" in files
    code = files["reach_target"]
    assert _is_valid_python(code), f"Invalid Python:\n{code}"
    assert "def verify_reach_target(" in code


def test_reward_generator_produces_one_file_per_task():
    from forge.compiler.generators.reward import RewardGenerator
    files = RewardGenerator().generate(_counter_input())
    assert "reach_target" in files
    code = files["reach_target"]
    assert _is_valid_python(code), f"Invalid Python:\n{code}"
    assert "def compute_reach_target_reward(" in code


def test_initial_state_generator_produces_valid_python():
    from forge.compiler.generators.initial_state import InitialStateGenerator
    code = InitialStateGenerator().generate(_counter_input())
    assert _is_valid_python(code), f"Invalid Python:\n{code}"
    assert "class CounterEnvInitialStateFactory" in code


def test_gym_wrapper_generator_produces_valid_python():
    from forge.compiler.generators.gym_wrapper import GymWrapperGenerator
    code = GymWrapperGenerator().generate(_counter_input())
    assert _is_valid_python(code), f"Invalid Python:\n{code}"
    assert "def build_counter_env_env(" in code
    assert 'te.register("increment"' in code


def test_test_suite_generator_produces_valid_python():
    from forge.compiler.generators.test_suite import TestSuiteGenerator
    gen = TestSuiteGenerator()
    files = gen.generate(_counter_input())
    assert "test_determinism" in files
    assert "test_transitions" in files
    assert "test_verifiers" in files
    for name, code in files.items():
        assert _is_valid_python(code), f"Invalid Python in {name}:\n{code}"


def test_gym_wrapper_contains_customization_loader():
    from forge.compiler.generators.gym_wrapper import GymWrapperGenerator
    code = GymWrapperGenerator().generate(_counter_input())
    assert "CustomizationLoader" in code
    assert "from forge.customization.loader import CustomizationLoader" in code
    assert "_pkg_dir = Path(__file__).parent" in code


def test_package_builder_custom_stubs_include_hooks_import():
    from forge.compiler.package_builder import PackageBuilder
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = PackageBuilder(Path(tmpdir)).build(_counter_input())
        transitions_stub = (pkg_dir / "custom" / "transitions.py").read_text()
        assert "override_transition" in transitions_stub
        config_stub = (pkg_dir / "custom" / "config.yaml").read_text()
        assert "base_success" in config_stub


def test_verifier_template_uses_exact_state_verifier():
    from forge.compiler.generators.verifier import VerifierGenerator
    ci = CompilerInput(
        project_name="test_env",
        domain="test",
        entities=[EntityDef(name="item", fields=[FieldDef(name="id", type="string")])],
        actions=[ActionDef(name="use", params=[ActionParam(name="id", type="string")])],
        tasks=[TaskTemplate(
            name="my_task",
            description="test",
            success_conditions=[SuccessCondition(type="state_check", expression="x > 0")],
        )],
    )
    code = VerifierGenerator().generate(ci)["my_task"]
    assert "ExactStateVerifier" in code
    assert "from forge.runtime.verifiers import" in code
    assert _is_valid_python(code)


def test_verifier_template_uses_event_verifier():
    from forge.compiler.generators.verifier import VerifierGenerator
    ci = CompilerInput(
        project_name="test_env",
        domain="test",
        entities=[EntityDef(name="item", fields=[FieldDef(name="id", type="string")])],
        actions=[ActionDef(name="use", params=[ActionParam(name="id", type="string")])],
        tasks=[TaskTemplate(
            name="my_task",
            description="test",
            success_conditions=[SuccessCondition(type="event_check", expression="item_used")],
        )],
    )
    code = VerifierGenerator().generate(ci)["my_task"]
    assert "EventVerifier" in code
    assert _is_valid_python(code)


def test_verifier_template_uses_semantic_verifier():
    from forge.compiler.generators.verifier import VerifierGenerator
    ci = CompilerInput(
        project_name="test_env",
        domain="test",
        entities=[EntityDef(name="item", fields=[FieldDef(name="id", type="string")])],
        actions=[ActionDef(name="use", params=[ActionParam(name="id", type="string")])],
        tasks=[TaskTemplate(
            name="my_task",
            description="test",
            success_conditions=[SuccessCondition(
                type="semantic_check", expression="reply_field",
                rubric="Must acknowledge the issue"
            )],
        )],
    )
    code = VerifierGenerator().generate(ci)["my_task"]
    assert "SemanticVerifier" in code
    assert "Must acknowledge the issue" in code
    assert 'mode="mock"' in code
    assert _is_valid_python(code)


def test_reward_template_has_five_components():
    from forge.compiler.generators.reward import RewardGenerator
    ci = _counter_input()
    code = RewardGenerator().generate(ci)["reach_target"]
    assert "task_success_reward" in code
    assert "policy_compliance_reward" in code
    assert "semantic_quality_reward" in code
    assert "action_cost" in code
    assert "invalid_action_penalty" in code
    assert "load_config" in code
    assert _is_valid_python(code)
