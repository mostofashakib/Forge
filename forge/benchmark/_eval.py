from __future__ import annotations


def evaluate_on_suite(model_path: str, suite: str) -> dict:
    """Evaluate a fine-tuned model zero-shot on a benchmark suite.

    Not yet implemented — requires the suite-specific evaluation harness.
    Install the eval harness and implement this function before running
    'forge benchmark eval' or 'forge benchmark transfer'.
    """
    raise NotImplementedError(
        f"evaluate_on_suite for suite='{suite}' is not yet implemented. "
        "Integrate the appropriate eval harness (e.g. WebArena/WorkArena) and "
        "implement this function in forge/benchmark/_eval.py."
    )
