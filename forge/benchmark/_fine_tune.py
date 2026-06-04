from __future__ import annotations
from pathlib import Path


def fine_tune_model(
    base_model: str,
    data_dir: Path,
    output_dir: Path,
    max_steps: int = 1000,
) -> str:
    """Fine-tune base_model on SFT data in data_dir.

    Not yet implemented — requires transformers + trl + a GPU node.
    Implement this function before running 'forge benchmark transfer'.

    Returns the path to the fine-tuned model checkpoint.
    """
    raise NotImplementedError(
        "fine_tune_model is not yet implemented. "
        "Install trl + transformers on a GPU node and implement this function "
        "in forge/benchmark/_fine_tune.py."
    )
