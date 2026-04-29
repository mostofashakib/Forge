import ast
import pytest
from forge.extraction.schemas import (
    CompilerInput, EntityDef, FieldDef, ActionDef, ActionParam,
    TaskTemplate, SuccessCondition,
)
from forge.compiler.generators.adversarial_test import AdversarialTestGenerator


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _ci_with_condition(cond_type: str, expression: str, rubric: str = "") -> CompilerInput:
    return CompilerInput(
        project_name="test_env",
        domain="test",
        entities=[EntityDef(name="item", fields=[FieldDef(name="id", type="string")])],
        actions=[ActionDef(name="use", params=[ActionParam(name="id", type="string")])],
        tasks=[TaskTemplate(
            name="my_task",
            description="test task",
            success_conditions=[SuccessCondition(
                type=cond_type, expression=expression, rubric=rubric
            )],
        )],
    )


def test_generates_state_check_adversarial():
    gen = AdversarialTestGenerator()
    result = gen.generate(_ci_with_condition("state_check", "x > 0"))
    assert "my_task_adversarial" in result
    code = result["my_task_adversarial"]
    assert "test_my_task_adversarial_state_1" in code
    assert "not result.passed" in code
    assert _is_valid_python(code)


def test_generates_event_check_adversarial():
    gen = AdversarialTestGenerator()
    result = gen.generate(_ci_with_condition("event_check", "ticket_resolved"))
    code = result["my_task_adversarial"]
    assert "test_my_task_adversarial_event_1" in code
    assert "events=[]" in code
    assert _is_valid_python(code)


def test_generates_temporal_check_adversarial():
    gen = AdversarialTestGenerator()
    result = gen.generate(_ci_with_condition("temporal_check", "ask_for_id before offer_refund"))
    code = result["my_task_adversarial"]
    assert "test_my_task_adversarial_temporal_1" in code
    assert "offer_refund" in code
    assert "ask_for_id" in code
    assert _is_valid_python(code)


def test_generates_policy_check_adversarial():
    gen = AdversarialTestGenerator()
    result = gen.generate(_ci_with_condition("policy_check", "forbidden_action"))
    code = result["my_task_adversarial"]
    assert "test_my_task_adversarial_policy_1" in code
    assert "forbidden_action" in code
    assert _is_valid_python(code)


def test_generates_negative_check_adversarial():
    gen = AdversarialTestGenerator()
    result = gen.generate(_ci_with_condition("negative_check", "premature_close"))
    code = result["my_task_adversarial"]
    assert "test_my_task_adversarial_negative_1" in code
    assert "premature_close" in code
    assert _is_valid_python(code)


def test_generates_semantic_check_adversarial_with_skip():
    gen = AdversarialTestGenerator(llm_client=None)
    result = gen.generate(_ci_with_condition("semantic_check", "reply_text", rubric="Must be polite"))
    code = result["my_task_adversarial"]
    assert "test_my_task_adversarial_semantic_1" in code
    assert "skip" in code.lower()
    assert _is_valid_python(code)


def test_adversarial_tests_fail_the_verifier(tmp_path):
    """Generated adversarial tests must actually catch the adversarial pattern."""
    import sys, importlib, tempfile, subprocess, os
    from forge.compiler.package_builder import PackageBuilder

    ci = CompilerInput(
        project_name="adv_env",
        domain="test",
        entities=[EntityDef(name="item", fields=[FieldDef(name="id", type="string")])],
        actions=[ActionDef(name="use", params=[ActionParam(name="id", type="string")])],
        tasks=[TaskTemplate(
            name="adv_task",
            description="test",
            success_conditions=[SuccessCondition(type="event_check", expression="item_used")],
        )],
    )
    pkg_dir = PackageBuilder(tmp_path).build(ci)
    adversarial_file = pkg_dir / "tests" / "adv_task_adversarial.py"
    assert adversarial_file.exists()

    env_copy = {**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    result = subprocess.run(
        ["pytest", str(adversarial_file), "-v", "--no-header", "--tb=short"],
        capture_output=True, text=True, env=env_copy,
    )
    assert "passed" in result.stdout, f"Adversarial test did not pass:\n{result.stdout}\n{result.stderr}"
