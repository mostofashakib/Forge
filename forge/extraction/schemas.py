from __future__ import annotations
from pydantic import BaseModel, Field, field_validator

from forge.paths import validate_identifier


class _NamedModel(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        return validate_identifier(value, label="name")


class FieldDef(_NamedModel):
    type: str  # string, integer, boolean, enum, list, dict
    values: list[str] | None = None  # for enum type
    nullable: bool = False
    default: object | None = None


class EntityDef(_NamedModel):
    primary_key: str = "id"
    fields: list[FieldDef]

    @field_validator("primary_key")
    @classmethod
    def _valid_primary_key(cls, value: str) -> str:
        return validate_identifier(value, label="primary_key")


class ActionParam(_NamedModel):
    type: str
    values: list[str] | None = None
    nullable: bool = False
    default: object | None = None


class ActionDef(_NamedModel):
    params: list[ActionParam]
    mutates: list[str] = Field(default_factory=list)
    requires_permission: list[str] = Field(default_factory=list)


class PolicyRule(BaseModel):
    id: str
    condition: str
    forbidden_actions: list[str]
    description: str = ""


class SuccessCondition(BaseModel):
    type: str  # state_check, event_check, temporal_check, policy_check, semantic_check, negative_check
    expression: str
    rubric: str = ""
    description: str = ""


class TaskTemplate(_NamedModel):
    description: str
    success_conditions: list[SuccessCondition]
    failure_conditions: list[SuccessCondition] = Field(default_factory=list)


class PermissionRule(BaseModel):
    action: str
    roles: list[str] = Field(default_factory=list)


class CompilerInput(BaseModel):
    project_name: str
    domain: str
    entities: list[EntityDef]
    actions: list[ActionDef]
    policies: list[PolicyRule] = Field(default_factory=list)
    tasks: list[TaskTemplate]
    permissions: list[PermissionRule] = Field(default_factory=list)

    @field_validator("project_name")
    @classmethod
    def _valid_module_name(cls, value: str, info) -> str:
        return validate_identifier(value, label=info.field_name)
