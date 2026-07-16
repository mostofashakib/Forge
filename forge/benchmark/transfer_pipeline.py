from __future__ import annotations
import logging
import importlib.util
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TransferConfig:
    data_dir: Path
    base_model: str
    output_dir: Path
    eval_suite: str = "webArena"
    max_train_steps: int = 1000


@dataclass
class TransferResult:
    model_path: str
    eval_suite: str
    task_completion_rate: float
    success_at_1: float
    success_at_3: float
    num_eval_tasks: int


def run_transfer_pipeline(config: TransferConfig) -> TransferResult:
    """Fine-tune base_model on Forge data, evaluate zero-shot on eval_suite.

    Requires a GPU node and the transformers + trl packages.
    Run via: forge benchmark transfer --data <dir> --base-model <model>
    """
    if importlib.util.find_spec("trl") is None:
        raise RuntimeError(
            "transfer_pipeline requires 'trl' and 'transformers'. "
            "Install them on a GPU node: pip install trl transformers datasets"
        )

    logger.info("[transfer] Fine-tuning %s on data from %s", config.base_model, config.data_dir)
    sft_dir = config.data_dir / "sft_pairs"
    if not sft_dir.exists():
        raise FileNotFoundError(
            f"SFT data not found at {sft_dir}. Run 'forge benchmark run' first."
        )

    from forge.benchmark._fine_tune import fine_tune_model
    model_path = fine_tune_model(
        base_model=config.base_model,
        data_dir=sft_dir,
        output_dir=config.output_dir / "forge_ft",
        max_steps=config.max_train_steps,
    )

    logger.info("[transfer] Evaluating on %s", config.eval_suite)
    from forge.benchmark._eval import evaluate_on_suite
    result = evaluate_on_suite(
        model_path=model_path,
        suite=config.eval_suite,
    )

    return TransferResult(
        model_path=model_path,
        eval_suite=config.eval_suite,
        task_completion_rate=result["task_completion_rate"],
        success_at_1=result["success_at_1"],
        success_at_3=result["success_at_3"],
        num_eval_tasks=result["num_eval_tasks"],
    )
