"""Episodes must stay distinguishable in grading prompts and results.

Agent episode ids start with ``cep_{seed:08x}``, so an 8-character prefix is
the same for every seed below 65536.
"""
from __future__ import annotations

from backend.app.api import detect, evaluate
from backend.app.models import AgentEpisode

EPISODES = [
    AgentEpisode(id="cep_00000001_aaaa", total_steps=1, total_reward=0.1,
                 final_objective_score=0.0, termination_reason="done"),
    AgentEpisode(id="cep_00000002_bbbb", total_steps=1, total_reward=0.9,
                 final_objective_score=0.0, termination_reason="done"),
]


def test_evaluate_prompt_names_each_episode_uniquely():
    text = evaluate._build_trajectory_text([(ep, []) for ep in EPISODES])
    assert all(ep.id in text for ep in EPISODES)


def test_detect_prompt_names_each_episode_uniquely():
    text = detect._build_prompt(EPISODES, {})
    assert all(ep.id in text for ep in EPISODES)


def test_llm_scores_are_matched_by_id_not_position(monkeypatch):
    def reversed_llm(_requirements, episodes):
        # The model lists the second episode first.
        return evaluate._RewardEvalResult(
            reevaluations=[
                evaluate._RewardReevaluation(episode_id=EPISODES[1].id, new_score=0.8, reasoning="r"),
                evaluate._RewardReevaluation(episode_id=EPISODES[0].id, new_score=0.2, reasoning="r"),
            ],
            summary="s",
        )

    monkeypatch.setattr(evaluate, "_run_reward_eval", reversed_llm)
    result, per_method = evaluate._run_reward_eval_multi(
        "req", [(ep, []) for ep in EPISODES], ["llm"]
    )

    scores = {r.episode_id: r.new_score for r in result.reevaluations}
    assert scores == {EPISODES[0].id: 0.2, EPISODES[1].id: 0.8}
    assert per_method["llm"] == [0.2, 0.8]


def test_episode_the_llm_skipped_gets_no_invented_score(monkeypatch):
    def partial_llm(_requirements, _episodes):
        return evaluate._RewardEvalResult(
            reevaluations=[
                evaluate._RewardReevaluation(episode_id=EPISODES[1].id, new_score=0.8, reasoning="r"),
                evaluate._RewardReevaluation(episode_id="cep_unknown", new_score=1.0, reasoning="r"),
            ],
            summary="s",
        )

    monkeypatch.setattr(evaluate, "_run_reward_eval", partial_llm)
    result, _ = evaluate._run_reward_eval_multi("req", [(ep, []) for ep in EPISODES], ["llm"])

    assert [r.episode_id for r in result.reevaluations] == [EPISODES[1].id]
