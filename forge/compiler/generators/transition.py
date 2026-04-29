from __future__ import annotations
from forge.compiler.generators.base import BaseGenerator
from forge.extraction.schemas import CompilerInput


class TransitionGenerator(BaseGenerator):
    def generate(self, compiler_input: CompilerInput) -> dict[str, str]:
        return {
            action.name: self.render("transition.py.j2", action=action)
            for action in compiler_input.actions
        }
