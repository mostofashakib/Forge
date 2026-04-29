from __future__ import annotations
import os
from pathlib import Path
from forge.compiler.package_builder import PackageBuilder
from forge.compiler.validation_runner import ValidationRunner, ValidationResult
from forge.extraction.schemas import CompilerInput


def run_compilation(compiler_input: CompilerInput) -> tuple[Path, ValidationResult]:
    root = Path(os.environ.get("FORGE_GENERATED_ENVS_DIR", "generated_envs"))
    root.mkdir(parents=True, exist_ok=True)
    pkg_dir = PackageBuilder(root).build(compiler_input)
    result = ValidationRunner(root).run(pkg_dir)
    return pkg_dir, result
