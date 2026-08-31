"""Heavy training backends — the actual gradient updates.

Kept separate from the orchestration (mirroring `forge/benchmark/_fine_tune.py`)
so the trainer's data loading, reward mapping, gating, and checkpoint contract
are testable without a GPU. Each backend gates on its optional GPU dependencies
and imports them only when training starts.

These backends train Forge's own policy from graded experience via GRPO / DPO;
the internal held-out harness evaluates their checkpoint separately.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Protocol

_INSTALL_HINT = (
    "Install the training extra on a GPU node: "
    "uv sync --extra training"
)


class TrainingBackend(Protocol):
    """Runs the gradient update and returns the checkpoint's model path."""

    def train(self, base_model: str, examples: list, output_dir: Path, max_steps: int) -> str: ...


def _require_training_deps(*packages: str) -> None:
    for pkg in packages:
        if importlib.util.find_spec(pkg) is None:
            raise RuntimeError(
                f"policy training requires '{pkg}', which is not installed. {_INSTALL_HINT}"
            )


class GRPOBackend:
    """Clipped batch-GRPO update over completions sampled by the base policy.

    Forge exports contain already sampled and graded completions. Before any
    update, this backend records their token log-probabilities under the frozen
    behavior policy (``base_model``). Training then uses GRPO's clipped
    current/old probability ratio plus a sampled KL penalty. This is a finite
    batch update, so collectors must export rollouts from ``base_model``.
    """

    clip_epsilon = 0.2
    kl_beta = 0.01

    def train(self, base_model: str, examples: list, output_dir: Path, max_steps: int) -> str:
        _require_training_deps("transformers", "datasets", "torch")
        import torch
        import torch.nn.functional as functional
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        output_dir = Path(output_dir)
        model_dir = output_dir / "forge_policy"
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize(example: dict) -> dict:
            prompt_ids = tokenizer(example["prompt"], add_special_tokens=True)["input_ids"]
            completion_ids = tokenizer(
                example["completion"], add_special_tokens=False
            )["input_ids"]
            if tokenizer.eos_token_id is not None:
                completion_ids = [*completion_ids, tokenizer.eos_token_id]
            return {
                "input_ids": [*prompt_ids, *completion_ids],
                "labels": [-100] * len(prompt_ids) + completion_ids,
                "advantage": float(example["advantage"]),
            }

        rows = [
            {
                "prompt": example.prompt,
                "completion": example.completion,
                "advantage": example.advantage,
            }
            for example in examples
        ]
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model.eval()
        tokenized_rows = [tokenize(row) for row in rows]
        # Freeze the exact behavior-policy likelihoods before Trainer performs
        # any gradient update. They are the denominator in GRPO's policy ratio.
        with torch.no_grad():
            for row in tokenized_rows:
                input_ids = torch.tensor(
                    [row["input_ids"]], dtype=torch.long, device=model.device
                )
                labels = torch.tensor(
                    [row["labels"]], dtype=torch.long, device=model.device
                )
                outputs = model(input_ids=input_ids)
                row["old_logps"] = _selected_token_logps(
                    functional, outputs.logits, labels
                )[0].cpu().tolist()
        dataset = Dataset.from_list(tokenized_rows)

        class AdvantageCollator:
            def __call__(self, features: list[dict]) -> dict:
                labels = [feature["labels"] for feature in features]
                advantages = [feature["advantage"] for feature in features]
                old_logps = [feature["old_logps"] for feature in features]
                model_inputs = [
                    {"input_ids": feature["input_ids"]} for feature in features
                ]
                batch = tokenizer.pad(
                    model_inputs, padding=True, return_tensors="pt"
                )
                width = batch["input_ids"].shape[1]
                batch["labels"] = torch.tensor(
                    [label + [-100] * (width - len(label)) for label in labels],
                    dtype=torch.long,
                )
                batch["advantages"] = torch.tensor(advantages, dtype=torch.float32)
                batch["old_logps"] = torch.tensor(
                    [values + [0.0] * (width - 1 - len(values)) for values in old_logps],
                    dtype=torch.float32,
                )
                return batch

        class AdvantageTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
                advantages = inputs.pop("advantages")
                old_logps = inputs.pop("old_logps")
                labels = inputs["labels"]
                outputs = model(**inputs)
                shifted_labels = labels[:, 1:].contiguous()
                mask = shifted_labels.ne(-100)
                current_logps = _selected_token_logps(
                    functional, outputs.logits, labels
                )
                old_logps = old_logps.to(current_logps.device)
                advantages = advantages.to(current_logps.device).unsqueeze(1)
                loss = _clipped_grpo_loss(
                    torch,
                    current_logps,
                    old_logps,
                    advantages,
                    mask,
                    clip_epsilon=GRPOBackend.clip_epsilon,
                    kl_beta=GRPOBackend.kl_beta,
                )
                return (loss, outputs) if return_outputs else loss

        model.train()
        arguments = TrainingArguments(
            output_dir=str(output_dir / "trainer_state"),
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=min(8, max(1, len(rows))),
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = AdvantageTrainer(
            model=model,
            args=arguments,
            train_dataset=dataset,
            data_collator=AdvantageCollator(),
        )
        trainer.train()
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(model_dir)
        return str(model_dir)


def _selected_token_logps(functional, logits, labels):
    """Log-probability of each sampled completion token, aligned after shift."""
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = functional.log_softmax(shifted_logits, dim=-1)
    selected = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~mask, 0.0)


def _clipped_grpo_loss(
    torch,
    current_logps,
    old_logps,
    advantages,
    mask,
    *,
    clip_epsilon: float,
    kl_beta: float,
):
    """GRPO clipped surrogate with a sampled reverse-KL penalty."""
    log_ratio = current_logps - old_logps
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.minimum(unclipped, clipped)
    kl = (-log_ratio).exp() + log_ratio - 1.0
    token_loss = policy_loss + kl_beta * kl
    sequence_losses = (
        (token_loss * mask).sum(1) / mask.sum(1).clamp_min(1)
    )
    return sequence_losses.mean()


class DPOBackend:
    """DPO trainer over chosen/rejected preference examples (trl DPOTrainer)."""

    def train(self, base_model: str, examples: list, output_dir: Path, max_steps: int) -> str:
        _require_training_deps("trl", "transformers", "datasets", "torch")
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer

        output_dir = Path(output_dir)
        model_dir = output_dir / "forge_policy"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = Dataset.from_list(
            [
                {
                    "prompt": example.prompt,
                    "chosen": example.chosen,
                    "rejected": example.rejected,
                }
                for example in examples
            ]
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        arguments = DPOConfig(
            output_dir=str(output_dir / "trainer_state"),
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=min(8, max(1, len(examples))),
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )
        trainer = DPOTrainer(
            model=base_model,
            args=arguments,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(model_dir)
        return str(model_dir)
