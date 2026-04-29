from __future__ import annotations
from forge.compiler.generators.base import BaseGenerator
from forge.extraction.schemas import CompilerInput


class ActionSchemaGenerator(BaseGenerator):
    def generate(self, compiler_input: CompilerInput) -> str:
        return self.render("action_models.py.j2", actions=compiler_input.actions)
