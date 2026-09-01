"""The template library must stay domain-neutral.

A library that ships a nurse and a patient quietly pushes every environment
toward a hospital: an author building a warehouse, a bank, or a moderation
queue starts by fighting the defaults instead of using them. Templates describe
how someone behaves; the author supplies who they are.
"""
from __future__ import annotations

import re

import pytest

from forge.personas.archetypes import ARCHETYPES, archetype, archetype_ids

# Words that name a specific industry rather than a way of behaving. If one of
# these is the right word for a template, the template is a character, not a
# disposition.
#
# Matched on word boundaries, not as substrings: "afterwards" contains "ward"
# and "impatient" contains "patient", and both are ordinary behavioral English.
DOMAIN_WORDS = (
    "nurse", "clinic", "clinical", "patient", "doctor", "physician", "ward",
    "medication", "discharge", "chart", "hospital", "medical",
    "ticket", "invoice", "refund", "shipment", "warehouse", "loan",
)


def _domain_words_in(text: str) -> list[str]:
    return [w for w in DOMAIN_WORDS if re.search(rf"\b{w}\b", text)]


@pytest.mark.parametrize("archetype_id", archetype_ids())
def test_no_template_names_an_industry(archetype_id):
    spec = ARCHETYPES[archetype_id]
    text = " ".join([
        spec.profile.id,
        spec.profile.name,
        spec.profile.role,
        spec.profile.backstory,
        spec.profile.style,
        *spec.profile.goals,
    ]).lower()
    offenders = _domain_words_in(text)
    assert not offenders, (
        f"{archetype_id} mentions {offenders} — templates describe behavior, "
        "not a domain. Move the domain detail into the environment that uses it."
    )


def test_the_neutrality_check_would_catch_a_domain_specific_template():
    """False-positive guard: the scan above must not be vacuous."""
    text = "priya raman, charge nurse on a busy ward"
    assert _domain_words_in(text) == ["nurse", "ward"]


def test_the_neutrality_check_tolerates_ordinary_english():
    """False-POSITIVE guard: "afterwards" and "impatient" are not domains."""
    assert _domain_words_in("impatient, and unwilling to wait afterwards") == []


@pytest.mark.parametrize("archetype_id", archetype_ids())
def test_no_template_claims_a_role(archetype_id):
    """Role is the author's to fill: it is where the domain enters."""
    assert ARCHETYPES[archetype_id].profile.role == ""


@pytest.mark.parametrize("archetype_id", archetype_ids())
def test_every_template_ships_inert(archetype_id):
    assert ARCHETYPES[archetype_id].behavior.allowed_actions == []


@pytest.mark.parametrize("archetype_id", archetype_ids())
def test_every_template_describes_a_disposition(archetype_id):
    """A template with no backstory teaches a model nothing about behaving."""
    assert len(ARCHETYPES[archetype_id].profile.backstory) > 40


@pytest.mark.parametrize("archetype_id", archetype_ids())
def test_every_template_is_named_as_a_placeholder_not_a_person(archetype_id):
    """A human name would read as a decision already made."""
    name = ARCHETYPES[archetype_id].profile.name
    assert name
    assert name.lower().replace(" ", "_") == archetype_id


def test_the_library_covers_more_than_cooperative_behavior():
    """An environment of helpful colleagues teaches an agent very little."""
    difficult = {"unreliable_third_party", "corner_cutter", "impatient_escalator", "gatekeeper"}
    assert difficult <= set(archetype_ids())


# --- dressing a template for a specific world ------------------------------


def test_identity_can_be_supplied_when_binding():
    spec = archetype(
        "gatekeeper",
        persona_id="supervisor",
        name="Alan Whitmore",
        role="shift supervisor",
        backstory="Answerable for the whole shift.",
        goals=["Keep it auditable"],
        allowed_actions=["approve"],
    )
    assert spec.profile.id == "supervisor"
    assert spec.profile.name == "Alan Whitmore"
    assert spec.profile.role == "shift supervisor"
    assert spec.profile.goals == ["Keep it auditable"]
    assert spec.behavior.allowed_actions == ["approve"]


def test_the_disposition_survives_being_dressed():
    """Renaming a gatekeeper must not turn them into a pushover."""
    spec = archetype("gatekeeper", name="Alan", role="supervisor")
    assert spec.profile.traits.diligence == ARCHETYPES["gatekeeper"].profile.traits.diligence
    assert spec.behavior.latency_steps == ARCHETYPES["gatekeeper"].behavior.latency_steps


def test_dressing_does_not_mutate_the_library():
    archetype("gatekeeper", name="Alan", role="supervisor", allowed_actions=["approve"])
    assert ARCHETYPES["gatekeeper"].profile.name == "Gatekeeper"
    assert ARCHETYPES["gatekeeper"].profile.role == ""
    assert ARCHETYPES["gatekeeper"].behavior.allowed_actions == []


def test_omitted_identity_fields_keep_the_template_default():
    spec = archetype("gatekeeper", role="supervisor")
    assert spec.profile.name == "Gatekeeper"


def test_an_explicitly_empty_goal_list_is_honored():
    """Distinguishing "not supplied" from "deliberately none"."""
    assert archetype("gatekeeper", goals=[]).profile.goals == []
