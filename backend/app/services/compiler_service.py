from __future__ import annotations
from pathlib import Path
from forge.compiler.package_builder import PackageBuilder
from forge.compiler.validation_runner import ValidationRunner, ValidationResult
from forge.extraction.schemas import CompilerInput
from forge.settings import generated_envs_root


def run_compilation(compiler_input: CompilerInput) -> tuple[Path, ValidationResult]:
    root = generated_envs_root()
    root.mkdir(parents=True, exist_ok=True)
    pkg_dir = PackageBuilder(root).build(compiler_input)
    result = ValidationRunner(root).run(pkg_dir)
    return pkg_dir, result
