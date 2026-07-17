from __future__ import annotations
from dataclasses import dataclass, field
from forge.extraction.schemas import CompilerInput


@dataclass
class EnvGenContext:
    env_name: str
    description: str
    compiler_input: CompilerInput
    policy_requirements: str = ""
    reward_requirements: str = ""
    reference_urls: list[str] = field(default_factory=list)
    source_product_name: str = ""
    source_product_url: str = ""
