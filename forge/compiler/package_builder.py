from __future__ import annotations
from pathlib import Path
from forge.compiler.generators.state_model import StateModelGenerator
from forge.compiler.generators.action_schema import ActionSchemaGenerator
from forge.compiler.generators.transition import TransitionGenerator
from forge.compiler.generators.verifier import VerifierGenerator
from forge.compiler.generators.reward import RewardGenerator
from forge.compiler.generators.initial_state import InitialStateGenerator
from forge.compiler.generators.gym_wrapper import GymWrapperGenerator
from forge.compiler.generators.test_suite import TestSuiteGenerator
from forge.compiler.generators.adversarial_test import AdversarialTestGenerator
from forge.extraction.schemas import CompilerInput
from forge.paths import confined_path, validate_identifier

_CUSTOM_STUBS = {
    "transitions.py": (
        "# Override transitions here using @override_transition decorator\n"
        "from forge.customization.hooks import override_transition\n"
        "\n"
        "# Example:\n"
        "# @override_transition('my_action')\n"
        "# def custom_my_action(state, action, ctx):\n"
        "#     import copy\n"
        "#     new_state = copy.deepcopy(state)\n"
        "#     # your logic here\n"
        "#     from forge.runtime.transition import TransitionResult\n"
        "#     return TransitionResult(state=new_state, events=[])\n"
    ),
    "verifiers.py": (
        "# Override verifiers here using @verifier decorator\n"
        "from forge.customization.hooks import verifier\n"
        "\n"
        "# Example:\n"
        "# @verifier('my_task')\n"
        "# def custom_verifier(state, trajectory, task):\n"
        "#     from forge.runtime.verification import CheckResult, VerificationResult\n"
        "#     checks = [CheckResult(name='custom_check', passed=True, score=1.0)]\n"
        "#     return VerificationResult.from_checks('my_task', checks)\n"
    ),
    "rewards.py": (
        "# Override rewards here using @reward decorator\n"
        "from forge.customization.hooks import reward\n"
        "\n"
        "# Example:\n"
        "# @reward('my_task')\n"
        "# def custom_reward(state, trajectory, verifier_results, task=None):\n"
        "#     from forge.runtime.reward import RewardBreakdown, RewardComponent\n"
        "#     return RewardBreakdown(total_reward=1.0, components=[])\n"
    ),
    "observations.py": (
        "# Override observations here using @observation_transform decorator\n"
        "from forge.customization.hooks import observation_transform\n"
        "\n"
        "# Example:\n"
        "# @observation_transform('agent_view')\n"
        "# def agent_view(state, actor):\n"
        "#     return {k: v for k, v in state.items() if k != 'hidden_field'}\n"
    ),
    "policies.py": (
        "# Override policy rules here using @policy_rule decorator\n"
        "from forge.customization.hooks import policy_rule\n"
        "\n"
        "# Example:\n"
        "# @policy_rule('no_action_without_context')\n"
        "# def no_action_without_context(state, action, ctx):\n"
        "#     return True  # return True to allow, False to block\n"
    ),
    "config.yaml": (
        "reward:\n"
        "  base_success: 1.0\n"
        "  step_penalty: 0.01\n"
        "  policy_violation_penalty: 1.0\n"
        "  max_reward: 1.0\n"
        "  min_reward: -1.0\n"
        "  semantic_weight: 0.0\n"
        "  invalid_action_penalty: 0.5\n"
        "\n"
        "observation:\n"
        "  mode: full\n"
        "  actor_role: agent\n"
        "  visible_entities: []\n"
        "  hidden_entities: []\n"
    ),
}


class PackageBuilder:
    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root

    def build(self, compiler_input: CompilerInput) -> Path:
        project_name = validate_identifier(compiler_input.project_name, label="project_name")
        pkg = confined_path(self._output_root, project_name)
        pkg.mkdir(parents=True, exist_ok=True)

        _write(pkg / "__init__.py", "")
        _write(pkg / "state_models.py", StateModelGenerator().generate(compiler_input))
        _write(pkg / "action_models.py", ActionSchemaGenerator().generate(compiler_input))
        _write(pkg / "initial_state.py", InitialStateGenerator().generate(compiler_input))
        _write(pkg / "gym_wrapper.py", GymWrapperGenerator().generate(compiler_input))

        for subdir in ("transitions", "verifiers", "rewards"):
            (pkg / subdir).mkdir(exist_ok=True)
            _write(pkg / subdir / "__init__.py", "")

        for name, code in TransitionGenerator().generate(compiler_input).items():
            _write(pkg / "transitions" / f"{validate_identifier(name, label='action name')}.py", code)
        for name, code in VerifierGenerator().generate(compiler_input).items():
            _write(pkg / "verifiers" / f"{validate_identifier(name, label='task name')}.py", code)
        for name, code in RewardGenerator().generate(compiler_input).items():
            _write(pkg / "rewards" / f"{validate_identifier(name, label='task name')}.py", code)

        tests_dir = pkg / "tests"
        tests_dir.mkdir(exist_ok=True)
        _write(tests_dir / "__init__.py", "")
        for name, code in TestSuiteGenerator().generate(compiler_input).items():
            _write(tests_dir / f"{name}.py", code)
        for name, code in AdversarialTestGenerator().generate(compiler_input).items():
            _write(tests_dir / f"{name}.py", code)

        custom_dir = pkg / "custom"
        custom_dir.mkdir(exist_ok=True)
        _write(custom_dir / "__init__.py", "")
        for filename, stub in _CUSTOM_STUBS.items():
            dest = custom_dir / filename
            if not dest.exists():
                _write(dest, stub)

        return pkg


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
