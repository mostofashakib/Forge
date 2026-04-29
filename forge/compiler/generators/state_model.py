from __future__ import annotations
from forge.compiler.generators.base import BaseGenerator
from forge.extraction.schemas import CompilerInput


class StateModelGenerator(BaseGenerator):
    def generate(self, compiler_input: CompilerInput) -> str:
        return self.render("state_models.py.j2", entities=compiler_input.entities)
