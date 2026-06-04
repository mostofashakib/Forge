from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class FieldSpec(BaseModel):
    type: Literal["integer", "string", "array", "object", "boolean", "datetime"]
    volatile: bool = False
    derived_from: list[str] = []
    required: bool = True


class StateSchemaManifest(BaseModel):
    env_name: str
    fields: dict[str, FieldSpec]

    def stable_fields(self) -> set[str]:
        return {name for name, spec in self.fields.items() if not spec.volatile}

    def coverage_score(self, actual_state: dict) -> float:
        required = [name for name, spec in self.fields.items() if spec.required]
        if not required:
            return 1.0
        present = sum(1 for name in required if name in actual_state)
        return present / len(required)

    def missing_fields(self, actual_state: dict) -> list[str]:
        return [
            name for name, spec in self.fields.items()
            if spec.required and name not in actual_state
        ]

    def state_changed(self, before: dict, after: dict) -> bool:
        stable = self.stable_fields()
        for field in stable:
            if before.get(field) != after.get(field):
                return True
        return False
