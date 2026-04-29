from __future__ import annotations
from forge.compiler.generators.base import BaseGenerator
from forge.extraction.schemas import CompilerInput


class TestSuiteGenerator(BaseGenerator):
    def generate(self, compiler_input: CompilerInput) -> dict[str, str]:
        ctx = dict(
            project_name=compiler_input.project_name,
            entities=compiler_input.entities,
            actions=compiler_input.actions,
            tasks=compiler_input.tasks,
        )
        return {
            "test_determinism": self.render("test_determinism.py.j2", **ctx),
            "test_transitions": self.render("test_transitions.py.j2", **ctx),
            "test_verifiers": self.render("test_verifiers.py.j2", **ctx),
        }
