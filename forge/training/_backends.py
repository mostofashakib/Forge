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
    """Offline group-relative policy update over graded Forge completions.

    Forge exports contain completions that have already been sampled and
    graded. The update therefore applies their group-relative advantages to
    token-level causal-LM loss instead of sampling unrelated completions again.
    Positive-advantage trajectories are reinforced and negative-advantage
    trajectories are suppressed.
    """

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
        dataset = Dataset.from_list(rows).map(tokenize, remove_columns=list(rows[0]))

        class AdvantageCollator:
            def __call__(self, features: list[dict]) -> dict:
                labels = [feature["labels"] for feature in features]
                advantages = [feature["advantage"] for feature in features]
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
                return batch

        class AdvantageTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
                advantages = inputs.pop("advantages")
                labels = inputs["labels"]
                outputs = model(**inputs)
                logits = outputs.logits[:, :-1, :].contiguous()
                shifted_labels = labels[:, 1:].contiguous()
                token_losses = functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    shifted_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(shifted_labels)
                mask = shifted_labels.ne(-100)
                sequence_losses = (token_losses * mask).sum(1) / mask.sum(1).clamp_min(1)
                loss = (sequence_losses * advantages.to(sequence_losses.device)).mean()
                return (loss, outputs) if return_outputs else loss

        model = AutoModelForCausalLM.from_pretrained(base_model)
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
