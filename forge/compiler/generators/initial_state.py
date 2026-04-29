from __future__ import annotations
from forge.compiler.generators.base import BaseGenerator
from forge.extraction.schemas import CompilerInput


class InitialStateGenerator(BaseGenerator):
    def generate(self, compiler_input: CompilerInput) -> str:
        return self.render(
            "initial_state.py.j2",
            project_name=compiler_input.project_name,
            entities=compiler_input.entities,
        )
