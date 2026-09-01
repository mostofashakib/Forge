"""Reading and writing the `personas:` block."""
from __future__ import annotations

import pytest
import yaml

from forge.personas.config import PersonaConfigError, dump_population, load_population


def test_missing_block_yields_a_disabled_population():
    """Every environment written before personas existed must load unchanged."""
    assert load_population(None).enabled is False
    assert load_population({}).enabled is False


def test_a_full_block_round_trips():
    raw = {
        "enabled": True,
        "count": 3,
        "driver": "anthropic:claude-sonnet-5",
        "max_actions_per_step": 2,
        "seed": 99,
        "roster": [
            {
                "id": "nurse_priya",
                "name": "Priya Raman",
                "role": "charge nurse",
                "backstory": "Catches errors.",
                "goals": ["Keep patients safe"],
                "style": "Confirms details.",
                "knowledge": {"allergy": "penicillin"},
                "traits": {"responsiveness": 80, "diligence": 95},
                "behavior": {
                    "allowed_actions": ["post_message"],
                    "wake_on": ["page_nurse"],
                    "activity": 30,
                    "latency_steps": 1,
                    "cooldown_steps": 2,
                    "max_actions_per_episode": 5,
                },
            }
        ],
    }
    population = load_population(raw)
    reloaded = load_population(dump_population(population))
    assert reloaded == population


def test_dumped_block_is_valid_yaml():
    population = load_population(
        {"enabled": True, "roster": [{"id": "n", "name": "N"}]}
    )
    assert load_population(yaml.safe_load(yaml.safe_dump(dump_population(population))))


def test_an_archetype_supplies_everything_an_entry_omits():
    population = load_population(
        {"enabled": True, "roster": [{"archetype": "meticulous_checker"}]}
    )
    spec = population.roster[0]
    assert spec.profile.name == "Meticulous checker"
    assert spec.profile.traits.diligence == 95


def test_entry_fields_override_the_archetype_they_extend():
    population = load_population(
        {
            "enabled": True,
            "roster": [
                {
                    "archetype": "meticulous_checker",
                    "id": "priya",
                    "name": "Priya R.",
                    "traits": {"diligence": 10},
                }
            ],
        }
    )
    spec = population.roster[0]
    assert spec.profile.id == "priya"
    assert spec.profile.name == "Priya R."
    assert spec.profile.traits.diligence == 10
    # Untouched traits still come from the archetype.
    assert spec.profile.traits.responsiveness == 80


def test_an_unknown_archetype_names_the_ones_that_exist():
    with pytest.raises(PersonaConfigError, match="available"):
        load_population({"roster": [{"archetype": "space_pirate"}]})


def test_a_misspelled_trait_is_rejected_rather_than_silently_defaulted():
    with pytest.raises(PersonaConfigError, match="respnsiveness"):
        load_population(
            {"roster": [{"id": "n", "name": "N", "traits": {"respnsiveness": 80}}]}
        )


def test_a_misspelled_behavior_field_is_rejected():
    with pytest.raises(PersonaConfigError, match="allowed_action"):
        load_population(
            {"roster": [{"id": "n", "name": "N", "behavior": {"allowed_action": []}}]}
        )


def test_a_misspelled_top_level_field_is_rejected():
    with pytest.raises(PersonaConfigError, match="enabeld"):
        load_population({"enabeld": True})


def test_an_out_of_range_trait_is_rejected():
    with pytest.raises(PersonaConfigError):
        load_population(
            {"roster": [{"id": "n", "name": "N", "traits": {"diligence": 400}}]}
        )


def test_a_negative_latency_is_rejected():
    with pytest.raises(PersonaConfigError):
        load_population(
            {"roster": [{"id": "n", "name": "N", "behavior": {"latency_steps": -1}}]}
        )


def test_duplicate_roster_ids_are_rejected():
    with pytest.raises(PersonaConfigError, match="duplicate"):
        load_population(
            {
                "roster": [
                    {"id": "n", "name": "A"},
                    {"id": "n", "name": "B"},
                ]
            }
        )


def test_max_actions_per_step_below_one_is_rejected():
    with pytest.raises(PersonaConfigError):
        load_population({"max_actions_per_step": 0})


def test_a_non_mapping_block_is_rejected():
    with pytest.raises(PersonaConfigError, match="mapping"):
        load_population(["nurse"])


def test_dumped_entries_are_expanded_not_archetype_references():
    """A round trip must not depend on a library entry that may later change."""
    population = load_population(
        {"enabled": True, "roster": [{"archetype": "busy_expert"}]}
    )
    entry = dump_population(population)["roster"][0]
    assert "archetype" not in entry
    assert entry["name"] == "Busy expert"


def test_env_config_loads_personas_from_disk(tmp_path):
    from forge.customization.config import load_config

    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "reward": {"base_success": 1.0},
                "personas": {
                    "enabled": True,
                    "roster": [
                        {
                            "id": "nurse",
                            "name": "Nurse",
                            "behavior": {"allowed_actions": ["post_message"]},
                        }
                    ],
                },
            }
        )
    )
    config = load_config(tmp_path)
    assert config.personas.enabled
    assert config.personas.roster[0].behavior.allowed_actions == ["post_message"]


def test_env_config_without_a_personas_block_stays_disabled(tmp_path):
    from forge.customization.config import load_config

    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "config.yaml").write_text(yaml.safe_dump({"reward": {"base_success": 1.0}}))
    assert load_config(tmp_path).personas.enabled is False
