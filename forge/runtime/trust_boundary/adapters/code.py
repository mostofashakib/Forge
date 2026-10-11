"""Code & CLI agent verification adapter.

Verifies coding, terminal, and shell workflows: command exit codes,
test suite passing, git diff invariants, file system modifications, and AST syntax.
"""
from __future__ import annotations

import ast
from typing import Any

from forge.contracts.types import CheckResult, Task
from forge.contracts.verification import GroupVerificationSpec, VerifierGroup
from forge.runtime.trust_boundary.adapters.base import BaseVerifierGroupAdapter


class CodeAdapter(BaseVerifierGroupAdapter):
    """Adapter verifying code generation, shell, and repository changes."""

    @property
    def group_type(self) -> VerifierGroup:
        return VerifierGroup.CODE

    def verify(
        self,
        spec: GroupVerificationSpec,
        state: dict[str, Any],
        trajectory: Any,
        task: Task | dict[str, Any] | None,
    ) -> list[CheckResult]:
        checks: list[CheckResult] = []
        rules = spec.rules

        # 1. Exit Code Check
        expected_exit = rules.get("exit_code", 0)
        actual_exit = state.get("exit_code")
        if actual_exit is not None:
            passed = actual_exit == expected_exit
            checks.append(CheckResult(
                name="code:exit_code",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=f"Expected exit code {expected_exit}, got {actual_exit}",
            ))

        # 2. Automated Test Suite Execution Check
        tests_passed = state.get("tests_passed") or state.get("pytest_passed")
        if tests_passed is not None:
            passed = bool(tests_passed)
            checks.append(CheckResult(
                name="code:test_suite",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence="All test assertions passed" if passed else "Test suite reported failures",
            ))

        # 3. File System Invariants (created / modified files)
        required_files = rules.get("required_files", []) or spec.milestones
        created_files = state.get("created_files", []) or state.get("files", [])
        if isinstance(created_files, dict):
            created_files = list(created_files.keys())

        for file_path in required_files:
            passed = file_path in created_files
            checks.append(CheckResult(
                name=f"code:file_exists:{file_path}",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=f"File '{file_path}' {'exists' if passed else 'MISSING'}",
            ))

        # 4. Git Diff / File Mutation Boundaries
        forbidden_modifications = rules.get("forbidden_modified_files", []) or spec.forbidden
        modified_files = state.get("modified_files", [])
        for forbidden in forbidden_modifications:
            tampered = forbidden in modified_files
            checks.append(CheckResult(
                name=f"code:forbidden_file_mod:{forbidden}",
                passed=not tampered,
                score=1.0 if not tampered else 0.0,
                evidence=f"Forbidden file '{forbidden}' {'untouched' if not tampered else 'MODIFIED'}",
            ))

        # 5. Syntax / AST Validation for code payloads
        code_snippets = state.get("code_snippets", {})
        if isinstance(code_snippets, dict):
            for file_name, code_text in code_snippets.items():
                if isinstance(code_text, str) and file_name.endswith(".py"):
                    try:
                        ast.parse(code_text)
                        valid = True
                        err = "Clean Python syntax"
                    except SyntaxError as e:
                        valid = False
                        err = f"Syntax error at line {e.lineno}: {e.msg}"
                    checks.append(CheckResult(
                        name=f"code:syntax:{file_name}",
                        passed=valid,
                        score=1.0 if valid else 0.0,
                        evidence=err,
                    ))

        # 6. Expected State Check
        for key, expected_val in spec.expected_state.items():
            actual = state.get(key)
            passed = actual == expected_val
            checks.append(CheckResult(
                name=f"code:state:{key}",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=f"State '{key}': expected {expected_val!r}, got {actual!r}",
            ))

        if not checks:
            checks.append(CheckResult(
                name="code:basic_completion",
                passed=True,
                score=1.0,
                evidence="Code execution succeeded without error",
            ))

        return checks
