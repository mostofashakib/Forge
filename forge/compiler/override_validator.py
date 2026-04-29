from __future__ import annotations
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from forge.customization.hooks import clear_registry, get_registry
from forge.extraction.schemas import CompilerInput


@dataclass
class OverrideValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class OverrideValidator:
    def validate(self, pkg_dir: Path, compiler_input: CompilerInput) -> OverrideValidationResult:
        custom_dir = pkg_dir / "custom"
        if not custom_dir.exists():
            return OverrideValidationResult(valid=True)

        clear_registry()
        for py_file in sorted(custom_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            _import_file(py_file)

        registry = get_registry()
        errors: list[str] = []
        action_names = {a.name for a in compiler_input.actions}
        task_names = {t.name for t in compiler_input.tasks}

        for action_name in registry["transitions"]:
            if action_name not in action_names:
                errors.append(
                    f"@override_transition('{action_name}') references unknown action "
                    f"'{action_name}'. Available: {sorted(action_names)}"
                )
        for task_name in registry["verifiers"]:
            if task_name not in task_names:
                errors.append(
                    f"@verifier('{task_name}') references unknown task "
                    f"'{task_name}'. Available: {sorted(task_names)}"
                )
        for task_name in registry["rewards"]:
            if task_name not in task_names:
                errors.append(
                    f"@reward('{task_name}') references unknown task "
                    f"'{task_name}'. Available: {sorted(task_names)}"
                )

        return OverrideValidationResult(valid=len(errors) == 0, errors=errors)


def _import_file(path: Path) -> None:
    module_name = f"_forge_override_check_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        pass
