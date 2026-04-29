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
from forge.extraction.schemas import CompilerInput

_CUSTOM_STUBS = {
    "transitions.py": "# Override transitions here using @override_transition decorator\n",
    "verifiers.py": "# Override verifiers here using @verifier decorator\n",
    "rewards.py": "# Override rewards here using @reward decorator\n",
    "config.yaml": "# Custom reward and observation config\n",
}


class PackageBuilder:
    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root

    def build(self, compiler_input: CompilerInput) -> Path:
        pkg = self._output_root / compiler_input.project_name
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
            _write(pkg / "transitions" / f"{name}.py", code)
        for name, code in VerifierGenerator().generate(compiler_input).items():
            _write(pkg / "verifiers" / f"{name}.py", code)
        for name, code in RewardGenerator().generate(compiler_input).items():
            _write(pkg / "rewards" / f"{name}.py", code)

        tests_dir = pkg / "tests"
        tests_dir.mkdir(exist_ok=True)
        _write(tests_dir / "__init__.py", "")
        for name, code in TestSuiteGenerator().generate(compiler_input).items():
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
