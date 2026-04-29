from __future__ import annotations
from forge.compiler.generators.base import BaseGenerator
from forge.extraction.schemas import CompilerInput


class RewardGenerator(BaseGenerator):
    def generate(self, compiler_input: CompilerInput) -> dict[str, str]:
        return {
            task.name: self.render("reward.py.j2", task=task)
            for task in compiler_input.tasks
        }
