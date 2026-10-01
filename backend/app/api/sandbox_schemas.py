"""Request and response models for the sandbox API."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

BUILD_IN_PROGRESS = ("queued", "building")


class CreateSandboxRequest(BaseModel):
    env_name: str
    env_type: Literal["general", "cli", "browser", "premade:gmail", "premade:slack"] = "general"
    description: str = Field(default="", max_length=50_000)
    domain: str = "localhost"
    policy_requirements: str = Field(default="", max_length=20_000)
    reward_requirements: str = Field(default="", max_length=20_000)
    reference_urls: list[str] = Field(default_factory=list, max_length=5)
    use_user_researcher: bool = False
    # Generated environments are API-only unless a UI is explicitly requested.
    # Ignored for cli (already headless) and browser (inherently a UI).
    with_ui: bool = False
    source_product_name: str = Field(default="", max_length=200)
    source_product_url: str = Field(default="", max_length=2_000)
    ttl_days: int = Field(default=30, ge=1, le=365)
    # The simulated people who will share the environment with the agent. The
    # builder chooses who is in the world; nobody can be given an action here,
    # because the environment's actions do not exist until it is generated.
    personas: dict | None = None

    @field_validator("personas")
    @classmethod
    def validate_personas(cls, value: dict | None) -> dict | None:
        """Reject a malformed cast at the request, not mid-build.

        A build that fails twenty minutes in because a trait was out of range
        is a much worse error than a 422 here.
        """
        if value is None:
            return None
        from forge.personas.config import PersonaConfigError, dump_population, load_population

        try:
            population = load_population(value)
        except PersonaConfigError as exc:
            raise ValueError(str(exc)) from exc
        for spec in [*population.roster, *population.archetypes]:
            if spec.behavior.allowed_actions:
                raise ValueError(
                    f"persona '{spec.profile.id}' cannot be granted actions at "
                    "creation time — the environment's actions do not exist "
                    "yet. Choose what each person can do on the Simulated "
                    "People page once the build finishes."
                )
        return dump_population(population)

    @field_validator("env_name")
    @classmethod
    def validate_env_name(cls, v: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", v):
            raise ValueError(
                "env_name must start with a letter or digit and contain only "
                "letters, digits, underscores, and hyphens (no spaces)"
            )
        return v

    @model_validator(mode="after")
    def normalize_ui_selection(self) -> CreateSandboxRequest:
        """The UI toggle governs generated apps only.

        Premade replicas always ship a UI and browser sandboxes are a browser;
        a CLI sandbox is a shell with no page at all. Only "general" builds
        actually run (or skip) the UI specialist.
        """
        if self.env_type != "general":
            self.with_ui = self.env_type != "cli"
        return self

    @model_validator(mode="after")
    def validate_research_source(self) -> CreateSandboxRequest:
        self.source_product_name = self.source_product_name.strip()
        self.source_product_url = self.source_product_url.strip()
        if not self.use_user_researcher:
            self.source_product_name = ""
            self.source_product_url = ""
            return self
        if not re.fullmatch(r"https?://[^\s]+", self.source_product_url):
            raise ValueError(
                "source_product_url must be a valid http or https URL when user research is enabled"
            )
        if not self.source_product_name:
            parsed = urlparse(self.source_product_url)
            hostname = (parsed.hostname or "").removeprefix("www.")
            source_key = hostname.split(".")[0]
            if hostname in {"github.com", "gitlab.com", "bitbucket.org"}:
                path_parts = [part for part in parsed.path.split("/") if part]
                if path_parts:
                    source_key = path_parts[-1].removesuffix(".git")
            self.source_product_name = re.sub(r"[-_]+", " ", source_key).strip().title()
        return self


class SandboxResponse(BaseModel):
    id: str
    status: str
    env_type: str = "general"
    has_ui: bool = True
    container_id: str | None = None
    container_port: int | None = None
    ttl_days: int
    expires_at: datetime
    created_at: datetime
    policy_requirements: str | None = None
    reward_requirements: str | None = None

    model_config = {"from_attributes": True}


class SandboxCapacityResponse(BaseModel):
    active_count: int
    limit: int
