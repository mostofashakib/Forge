from . import trajectories, rewards, verifier_results, sft_pairs, preference_pairs, grpo_rollouts, failure_dataset

WRITERS = {
    "trajectories": trajectories.write,
    "rewards": rewards.write,
    "verifier_results": verifier_results.write,
    "sft_pairs": sft_pairs.write,
    "preference_pairs": preference_pairs.write,
    "grpo_rollouts": grpo_rollouts.write,
    "failure_dataset": failure_dataset.write,
}
