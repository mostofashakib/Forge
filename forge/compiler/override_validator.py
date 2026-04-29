from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from forge.customization.hooks import clear_registry, get_registry, import_custom_file
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
        errors: list[str] = []
        for py_file in sorted(custom_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                import_custom_file(py_file, "_forge_override_check")
            except Exception as exc:
                errors.append(f"Failed to load '{py_file.name}': {exc}")

        registry = get_registry()
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
