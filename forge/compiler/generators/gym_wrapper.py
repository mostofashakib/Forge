from __future__ import annotations
from forge.compiler.generators.base import BaseGenerator
from forge.extraction.schemas import CompilerInput


class GymWrapperGenerator(BaseGenerator):
    def generate(self, compiler_input: CompilerInput) -> str:
        return self.render(
            "gym_wrapper.py.j2",
            project_name=compiler_input.project_name,
            domain=compiler_input.domain,
            actions=compiler_input.actions,
            tasks=compiler_input.tasks,
        )
