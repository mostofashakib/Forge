from __future__ import annotations
from pydantic import BaseModel, Field


class FieldDef(BaseModel):
    name: str
    type: str  # string, integer, boolean, enum, list, dict
    values: list[str] | None = None  # for enum type
    nullable: bool = False
    default: object | None = None


class EntityDef(BaseModel):
    name: str
    primary_key: str = "id"
    fields: list[FieldDef]


class ActionParam(BaseModel):
    name: str
    type: str
    values: list[str] | None = None
    nullable: bool = False
    default: object | None = None


class ActionDef(BaseModel):
    name: str
    params: list[ActionParam]
    mutates: list[str] = Field(default_factory=list)
    requires_permission: list[str] = Field(default_factory=list)


class PolicyRule(BaseModel):
    id: str
    condition: str
    forbidden_actions: list[str]
    description: str = ""


class SuccessCondition(BaseModel):
    type: str  # state_check, event_check, temporal_check, negative_check
    expression: str
    description: str = ""


class TaskTemplate(BaseModel):
    name: str
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
