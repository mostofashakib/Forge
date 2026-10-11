"""Data & Database agent verification adapter.

Verifies database, SQL, ETL, and analytics workflows: table row count invariants,
SQL query assertion outputs, schema constraints, and data integrity metrics.
"""
from __future__ import annotations

from typing import Any

from forge.contracts.types import CheckResult, Task
from forge.contracts.verification import GroupVerificationSpec, VerifierGroup
from forge.runtime.trust_boundary.adapters.base import BaseVerifierGroupAdapter


class DataAdapter(BaseVerifierGroupAdapter):
    """Adapter verifying database and data engineering agent trajectories."""

    @property
    def group_type(self) -> VerifierGroup:
        return VerifierGroup.DATA

    def verify(
        self,
        spec: GroupVerificationSpec,
        state: dict[str, Any],
        trajectory: Any,
        task: Task | dict[str, Any] | None,
    ) -> list[CheckResult]:
        checks: list[CheckResult] = []
        rules = spec.rules

        # 1. Table Row Counts / Table Invariants
        expected_counts = rules.get("table_row_counts", {})
        if isinstance(expected_counts, dict):
            tables = state.get("tables", {}) or state.get("database", {})
            for table_name, expected_count in expected_counts.items():
                table_data = tables.get(table_name)
                actual_count = len(table_data) if isinstance(table_data, (list, dict)) else state.get(f"{table_name}_count")
                passed = actual_count == expected_count
                checks.append(CheckResult(
                    name=f"data:row_count:{table_name}",
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    evidence=f"Table '{table_name}': expected {expected_count} rows, found {actual_count}",
                ))

        # 2. SQL Query Invariant Assertions
        query_assertions = rules.get("query_assertions", {})
        if isinstance(query_assertions, dict):
            query_results = state.get("query_results", {})
            for query_key, expected_val in query_assertions.items():
                actual_val = query_results.get(query_key)
                passed = actual_val == expected_val
                checks.append(CheckResult(
                    name=f"data:query_assert:{query_key}",
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    evidence=f"Query '{query_key}': expected {expected_val!r}, got {actual_val!r}",
                ))

        # 3. Schema Invariants (Required tables / columns)
        required_tables = rules.get("required_tables", []) or spec.milestones
        existing_tables = state.get("tables", {}) or state.get("schema", {})
        if isinstance(existing_tables, dict):
            existing_table_names = list(existing_tables.keys())
        elif isinstance(existing_tables, list):
            existing_table_names = existing_tables
        else:
            existing_table_names = []

        for table in required_tables:
            passed = table in existing_table_names
            checks.append(CheckResult(
                name=f"data:table_exists:{table}",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=f"Table '{table}' {'exists' if passed else 'MISSING'}",
            ))

        # 4. Forbidden Mutations (Dropped tables or destructive queries)
        forbidden_dropped = rules.get("forbidden_dropped_tables", []) or spec.forbidden
        dropped_tables = state.get("dropped_tables", [])
        for table in forbidden_dropped:
            tampered = table in dropped_tables
            checks.append(CheckResult(
                name=f"data:forbidden_drop:{table}",
                passed=not tampered,
                score=1.0 if not tampered else 0.0,
                evidence=f"Table '{table}' was {'preserved' if not tampered else 'DROPPED'}",
            ))

        # 5. Expected State Keys
        for key, expected_val in spec.expected_state.items():
            actual = state.get(key)
            passed = actual == expected_val
            checks.append(CheckResult(
                name=f"data:state:{key}",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=f"State '{key}': expected {expected_val!r}, got {actual!r}",
            ))

        if not checks:
            checks.append(CheckResult(
                name="data:basic_completion",
                passed=True,
                score=1.0,
                evidence="Data pipeline executed cleanly",
            ))

        return checks
