from __future__ import annotations
from pathlib import Path
from forge.customization.hooks import clear_registry, get_registry, import_custom_file
from forge.runtime.reward import RewardEngine
from forge.runtime.transition import TransitionEngine
from forge.runtime.verifier import VerifierEngine


class CustomizationLoader:
    def __init__(self, pkg_dir: Path) -> None:
        self._custom_dir = pkg_dir / "custom"

    def apply(
        self,
        transition_engine: TransitionEngine,
        verifier_engine: VerifierEngine,
        reward_engine: RewardEngine,
    ) -> None:
        if not self._custom_dir.exists():
            return
        clear_registry()
        for py_file in sorted(self._custom_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            import_custom_file(py_file, "_forge_custom")
        registry = get_registry()
        for action_name, fn in registry["transitions"].items():
            transition_engine.register(action_name, fn)
        for task_name, fn in registry["verifiers"].items():
            verifier_engine.register(task_name, fn)
        for task_name, fn in registry["rewards"].items():
            reward_engine.register(task_name, fn)
