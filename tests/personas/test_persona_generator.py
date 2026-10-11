import json
import pytest

from forge.contracts.persona import PersonaSpec, PersonaPopulation, TRAIT_MIN, TRAIT_MAX
from forge.personas.generator import PersonaGenerator, PersonaLLMClient


class MockLLMClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response_text


def test_deterministic_generation_repeatability():
    generator = PersonaGenerator()
    p1 = generator.generate_persona(role="Senior Auditor", context="Tax Compliance", seed=42)
    p2 = generator.generate_persona(role="Senior Auditor", context="Tax Compliance", seed=42)

    assert p1.profile.id == p2.profile.id
    assert p1.profile.name == p2.profile.name
    assert p1.profile.traits.responsiveness == p2.profile.traits.responsiveness
    assert p1.profile.traits.diligence == p2.profile.traits.diligence
    assert p1.profile.style == p2.profile.style


def test_deterministic_generation_different_seeds():
    generator = PersonaGenerator()
    p1 = generator.generate_persona(role="Support Agent", seed=10)
    p2 = generator.generate_persona(role="Support Agent", seed=99)

    assert p1.profile.name != p2.profile.name or p1.profile.traits != p2.profile.traits


def test_traits_within_contracts():
    generator = PersonaGenerator()
    persona = generator.generate_persona(role="Physician Assistant", context="Emergency Department", seed=123)

    traits = persona.profile.traits
    for trait_name in ["responsiveness", "initiative", "verbosity", "diligence", "formality", "patience"]:
        val = getattr(traits, trait_name)
        assert TRAIT_MIN <= val <= TRAIT_MAX


def test_allowed_actions_propagated():
    generator = PersonaGenerator()
    allowed = ["send_email", "query_database"]
    persona = generator.generate_persona(role="Billing Clerk", allowed_actions=allowed, seed=7)

    assert persona.behavior.allowed_actions == allowed


def test_generate_population():
    generator = PersonaGenerator()
    roles = ["Triage Nurse", "Attending Physician", "Radiologist"]
    pop = generator.generate_population(
        scenario="Level 1 Trauma Center",
        roles=roles,
        count=5,
        allowed_actions=["page_staff", "order_labs"],
        seed=100,
    )

    assert isinstance(pop, PersonaPopulation)
    assert pop.enabled is True
    assert pop.count == 5
    assert len(pop.roster) == 5
    for p in pop.roster:
        assert p.behavior.allowed_actions == ["page_staff", "order_labs"]


def test_llm_backed_generation():
    mock_payload = {
        "id": "dr_chen",
        "name": "Dr. Eleanor Chen",
        "role": "Chief Surgeon",
        "backstory": "20 years of experience, dislikes micromanagement.",
        "goals": ["Ensure surgical safety", "Minimize operation turnaround time"],
        "traits": {
            "responsiveness": 85,
            "initiative": 75,
            "verbosity": 40,
            "diligence": 95,
            "formality": 80,
            "patience": 30,
        },
        "knowledge": {"patient_allergy": "Penicillin"},
        "style": "Extremely direct. Never uses pleasantries.",
    }
    mock_client = MockLLMClient(response_text=f"```json\n{json.dumps(mock_payload)}\n```")
    generator = PersonaGenerator(llm_client=mock_client)

    persona = generator.generate_persona(role="Chief Surgeon", context="Cardiology Clinic")

    assert persona.profile.name == "Dr. Eleanor Chen"
    assert persona.profile.id == "dr_chen"
    assert persona.profile.traits.responsiveness == 85
    assert persona.profile.traits.diligence == 95
    assert persona.profile.knowledge == {"patient_allergy": "Penicillin"}
    assert "Cardiology Clinic" in mock_client.prompts[0]


def test_llm_malformed_json_falls_back():
    broken_client = MockLLMClient(response_text="Sorry, I cannot produce JSON right now.")
    generator = PersonaGenerator(llm_client=broken_client)

    # Should not raise exception, falls back to deterministic rule generator
    persona = generator.generate_persona(role="Compliance Officer", seed=42)
    assert isinstance(persona, PersonaSpec)
    assert persona.profile.role == "Compliance Officer"
