"""Dynamic persona generation for multi-turn simulated agent environments.

Provides LLM-driven and deterministic persona synthesis to create realistic,
context-specific human participants (colleagues, clients, reviewers) on demand.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from forge.contracts.persona import (
    PersonaBehavior,
    PersonaPopulation,
    PersonaProfile,
    PersonaSpec,
    PersonaTraits,
)


@runtime_checkable
class PersonaLLMClient(Protocol):
    """Protocol for LLM completion providers used during persona synthesis."""

    def complete(self, prompt: str) -> str:
        """Return raw model completion text for the given prompt."""
        ...


_GENERATION_SYSTEM_PROMPT = """You are an expert behavioral simulator designer.
Your task is to generate realistic, non-generic human personas that inhabit a simulation alongside an AI agent.
Avoid generic or robotic archetypes. Ground the persona with authentic communication quirks, specific private knowledge, and realistic workplace flaws.

Output ONLY valid JSON matching this schema:
{
  "id": "short_unique_id",
  "name": "Full Name",
  "role": "Job Title or Role",
  "backstory": "1-2 sentences on their context and history",
  "goals": ["primary objective", "secondary priority"],
  "traits": {
    "responsiveness": 0-100,
    "initiative": 0-100,
    "verbosity": 0-100,
    "diligence": 0-100,
    "formality": 0-100,
    "patience": 0-100
  },
  "knowledge": {
    "key_fact_1": "private information only this persona knows"
  },
  "style": "Exact communication instructions, e.g. 'Short fragments, never uses greetings, frequently typos'."
}
"""


class PersonaGenerator:
    """Synthesizes context-grounded PersonaSpec and PersonaPopulation instances."""

    def __init__(self, llm_client: PersonaLLMClient | None = None) -> None:
        self._llm_client = llm_client

    def generate_persona(
        self,
        role: str,
        context: str = "",
        allowed_actions: list[str] | None = None,
        seed: int | None = None,
    ) -> PersonaSpec:
        """Generate a single PersonaSpec for a role within an environment context."""
        actions = allowed_actions or []
        if self._llm_client is not None:
            try:
                return self._generate_with_llm(
                    role=role,
                    context=context,
                    allowed_actions=actions,
                )
            except Exception as exc:
                from forge.runtime.error_handling import get_error_handler
                get_error_handler().capture(
                    exc,
                    origin="persona_generator",
                    severity="WARNING",
                    context={"role": role, "context": context},
                    reraise=False,
                )

        return self._generate_deterministic(
            role=role,
            context=context,
            allowed_actions=actions,
            seed=seed,
        )

    def generate_population(
        self,
        scenario: str,
        roles: list[str],
        count: int | None = None,
        allowed_actions: list[str] | None = None,
        seed: int | None = None,
    ) -> PersonaPopulation:
        """Generate a full PersonaPopulation cast for a scenario."""
        actions = allowed_actions or []
        target_count = count if count is not None else len(roles)
        roster: list[PersonaSpec] = []

        rng = random.Random(seed) if seed is not None else random.Random()

        for idx, role in enumerate(roles):
            role_seed = rng.randint(0, 1_000_000) if seed is not None else None
            persona = self.generate_persona(
                role=role,
                context=scenario,
                allowed_actions=actions,
                seed=role_seed,
            )
            roster.append(persona)

        # If more personas are requested than roles provided, synthesize variations
        while len(roster) < target_count:
            idx = len(roster)
            base_role = roles[idx % len(roles)] if roles else "Team Member"
            role_seed = rng.randint(0, 1_000_000) if seed is not None else None
            persona = self.generate_persona(
                role=f"{base_role} #{idx + 1}",
                context=scenario,
                allowed_actions=actions,
                seed=role_seed,
            )
            roster.append(persona)

        return PersonaPopulation(
            enabled=True,
            count=target_count,
            roster=roster[:target_count],
            seed=seed,
        )

    def _generate_with_llm(
        self,
        role: str,
        context: str,
        allowed_actions: list[str],
    ) -> PersonaSpec:
        assert self._llm_client is not None
        prompt = (
            f"{_GENERATION_SYSTEM_PROMPT}\n\n"
            f"Environment Context: {context or 'Modern corporate workplace'}\n"
            f"Requested Role: {role}\n"
            "Generate JSON for this persona:"
        )
        response = self._llm_client.complete(prompt)
        parsed = self._extract_json(response)
        profile = self._build_profile_from_dict(parsed, fallback_role=role)

        behavior = PersonaBehavior(
            allowed_actions=allowed_actions,
            activity=max(10, min(90, profile.traits.initiative)),
            latency_steps=max(0, 5 - (profile.traits.responsiveness // 20)),
            cooldown_steps=1 if profile.traits.patience > 40 else 0,
        )
        return PersonaSpec(profile=profile, behavior=behavior)

    def _generate_deterministic(
        self,
        role: str,
        context: str,
        allowed_actions: list[str],
        seed: int | None,
    ) -> PersonaSpec:
        """Deterministic rule-based generator ensuring zero-dependency repeatability."""
        seed_value = seed
        if seed_value is None:
            # Deterministic hash from role + context
            key = f"{role}:{context}"
            seed_value = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)

        rng = random.Random(seed_value)

        first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Casey", "Riley", "Chris"]
        last_names = ["Chen", "Patel", "Novak", "Smith", "Kim", "Garcia", "Wong", "Davis"]
        name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
        clean_id = re.sub(r"[^a-z0-9_]", "", f"{role.lower().replace(' ', '_')}_{name.lower().replace(' ', '_')}")[:24]

        # Vary traits coherently based on role stereotypes and RNG jitter
        responsiveness = max(10, min(95, 50 + rng.randint(-30, 30)))
        initiative = max(10, min(95, 40 + rng.randint(-25, 35)))
        verbosity = max(15, min(90, 50 + rng.randint(-25, 25)))
        diligence = max(20, min(95, 60 + rng.randint(-30, 30)))
        formality = max(10, min(90, 50 + rng.randint(-30, 30)))
        patience = max(10, min(90, 50 + rng.randint(-30, 30)))

        traits = PersonaTraits(
            responsiveness=responsiveness,
            initiative=initiative,
            verbosity=verbosity,
            diligence=diligence,
            formality=formality,
            patience=patience,
        )

        styles = [
            "Terse and focused. Prefers bullet points and skips salutations.",
            "Friendly and conversational, uses emojis occasionally.",
            "Extremely formal and procedural, cites company policy when possible.",
            "Fast-paced, asks urgent follow-up questions.",
            "Quiet and methodical, provides comprehensive detail only when asked directly.",
        ]
        style = rng.choice(styles)

        profile = PersonaProfile(
            id=clean_id,
            name=name,
            role=role,
            backstory=f"{role} working in {context or 'the organization'} for {rng.randint(1, 8)} years.",
            goals=[f"Complete {role.lower()} tasks efficiently", "Minimize interruptions"],
            traits=traits,
            knowledge={"domain_context": context, "assigned_role": role},
            style=style,
        )

        behavior = PersonaBehavior(
            allowed_actions=allowed_actions,
            activity=max(10, min(80, initiative)),
            latency_steps=max(0, 4 - (responsiveness // 25)),
            cooldown_steps=1 if patience > 40 else 0,
        )

        return PersonaSpec(profile=profile, behavior=behavior)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract first valid JSON object from LLM completion string."""
        text = text.strip()
        # Direct parse attempt
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Match markdown block ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Match outer braces
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        raise ValueError(f"Could not parse valid JSON from text: {text[:100]}...")

    @staticmethod
    def _build_profile_from_dict(data: dict[str, Any], fallback_role: str) -> PersonaProfile:
        traits_dict = data.get("traits") or {}
        traits = PersonaTraits(
            responsiveness=max(0, min(100, int(traits_dict.get("responsiveness", 50)))),
            initiative=max(0, min(100, int(traits_dict.get("initiative", 30)))),
            verbosity=max(0, min(100, int(traits_dict.get("verbosity", 50)))),
            diligence=max(0, min(100, int(traits_dict.get("diligence", 70)))),
            formality=max(0, min(100, int(traits_dict.get("formality", 50)))),
            patience=max(0, min(100, int(traits_dict.get("patience", 50)))),
        )

        raw_id = str(data.get("id") or fallback_role.lower().replace(" ", "_"))
        clean_id = re.sub(r"[^a-z0-9_]", "", raw_id)[:32] or "persona_agent"

        return PersonaProfile(
            id=clean_id,
            name=str(data.get("name") or "Simulated Colleague"),
            role=str(data.get("role") or fallback_role),
            backstory=str(data.get("backstory") or ""),
            goals=list(data.get("goals") or []),
            traits=traits,
            knowledge=dict(data.get("knowledge") or {}),
            style=str(data.get("style") or ""),
        )
