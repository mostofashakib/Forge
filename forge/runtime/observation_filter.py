from __future__ import annotations
from pydantic import BaseModel, Field


class RolePermissions(BaseModel):
    can_see: list[str] = Field(default_factory=list)
    cannot_see: list[str] = Field(default_factory=list)


class RBACConfig(BaseModel):
    roles: dict[str, RolePermissions] = Field(default_factory=dict)


class ObservationFilter:
    def __init__(
        self,
        rbac_config: RBACConfig | None = None,
        role: str | None = None,
    ) -> None:
        self._config = rbac_config
        self._role = role

    def filter(self, obs: dict) -> dict:
        if self._config is None or self._role is None:
            return obs
        permissions = self._config.roles.get(self._role)
        if permissions is None:
            return obs

        result = dict(obs)

        if permissions.can_see:
            result = {k: v for k, v in result.items() if k in permissions.can_see}

        for key in permissions.cannot_see:
            result.pop(key, None)

        return result
