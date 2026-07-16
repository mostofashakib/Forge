from __future__ import annotations
from dataclasses import dataclass
from forge.extraction.schemas import CompilerInput


@dataclass
class EnvGenContext:
    env_name: str
    description: str
    compiler_input: CompilerInput
    policy_requirements: str = ""
    reward_requirements: str = ""
