"""Specialists must name the contracts, not describe them in prose."""
from __future__ import annotations

import inspect
import re

from forge.contracts.reward import Rubric
from forge.envgen.agents.app_generator import AppGeneratorPrompts
from forge.envgen.agents.state_bridge import StateBridgePrompts
from forge.envgen.agents.reward import RewardPrompts


def test_the_backend_prompt_names_the_state_manager_contract():
    assert "StateManager" in AppGeneratorPrompts.BACKEND
    assert "reset_state" in AppGeneratorPrompts.BACKEND


def test_the_state_bridge_prompt_names_the_environment_facade():
    assert "Environment" in StateBridgePrompts.SYSTEM
    assert "forge.contracts" in StateBridgePrompts.SYSTEM


def test_the_reward_prompt_names_the_rubric_contract():
    assert "Rubric" in RewardPrompts.SYSTEM
    assert "def score(" in RewardPrompts.SYSTEM


def test_backend_prompt_state_manager_citation_is_inside_the_state_block_not_pasted_elsewhere():
    """A false-positive guard: the citation must sit within the
    STATE-MANAGEMENT CLASS block, not merely appear anywhere in the prompt
    (e.g. pasted at the very end)."""
    marker_index = AppGeneratorPrompts.BACKEND.index("STATE-MANAGEMENT CLASS")
    state_manager_index = AppGeneratorPrompts.BACKEND.index(
        "StateManager", marker_index
    )
    # The citation should appear reasonably close to the marker, within the
    # same block, not off in some unrelated later section.
    next_section_markers = ["DETERMINISM CONTRACT"]
    next_section_index = min(
        AppGeneratorPrompts.BACKEND.index(m, marker_index)
        for m in next_section_markers
        if m in AppGeneratorPrompts.BACKEND[marker_index:]
    )
    assert marker_index < state_manager_index < next_section_index


def test_reward_prompt_does_not_mention_unrelated_contracts():
    """Negative case: the reward specialist should not be citing contracts
    that belong to other concerns — the citation is targeted, not a blanket
    paste of every contract name into every prompt."""
    assert "TransitionHandler" not in RewardPrompts.SYSTEM
    assert "StateManager" not in RewardPrompts.SYSTEM


def test_reward_prompt_signature_matches_the_real_rubric_score_signature():
    """Accuracy guard: introspect the real Rubric.score signature and assert
    every one of its parameter names appears in the prompt's `def score(`
    line, so the prompt cannot silently drift from the contract.

    Uses a word-boundary regex rather than plain substring containment: a
    substring check is blind to one rename direction — if `verifier_results`
    were renamed to `results`, the new name `results` is itself a substring
    of the stale prompt text `verifier_results`, so plain `in` would keep
    passing against a prompt that no longer matches the real contract.
    """
    real_params = list(inspect.signature(Rubric.score).parameters)
    assert real_params[0] == "self"

    system = RewardPrompts.SYSTEM
    def_index = system.index("def score(")
    line_end = system.index("\n", def_index)
    score_line = system[def_index:line_end]

    for param in real_params:
        if param == "self":
            continue
        assert re.search(rf"\b{re.escape(param)}\b", score_line), (
            f"prompt's `def score(` line is missing real parameter {param!r}: "
            f"{score_line!r}"
        )
